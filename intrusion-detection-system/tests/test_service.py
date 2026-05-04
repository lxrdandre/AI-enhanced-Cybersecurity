from __future__ import annotations

import numpy as np

from app.service import InferenceService


class _IdentityScaler:
    """Represent identity scaler state and behavior."""
    def transform(self, values):
        """Return the input transformed as an identity array."""
        return values


class _TargetEncoder:
    """Represent target encoder state and behavior."""
    classes_ = np.array(["normal", "xss"])


class _DosTargetEncoder:
    """Represent dos target encoder state and behavior."""
    classes_ = np.array(["normal", "dos_ddos"])


class _ReversedTargetEncoder:
    """Represent reversed target encoder state and behavior."""
    classes_ = np.array(["xss", "normal"])


class _Model:
    """Represent model state and behavior."""
    input_shape = (None, 1)

    def __init__(self, probs):
        """Initialize the model instance."""
        self.probs = np.asarray([probs], dtype=np.float32)

    def predict(self, x, verbose=0):
        """Run model inference for submitted records."""
        return np.repeat(self.probs, len(x), axis=0)


class _RecordingModel(_Model):
    """Represent recording model state and behavior."""
    def __init__(self, probs):
        """Initialize the recording model instance."""
        super().__init__(probs)
        self.last_x = None

    def predict(self, x, verbose=0):
        """Run model inference for submitted records."""
        self.last_x = np.asarray(x)
        return super().predict(x, verbose=verbose)


class _Router:
    """Represent router state and behavior."""
    classes_ = np.array([0, 1])

    def __init__(self):
        """Initialize the router instance."""
        self.calls = 0

    def predict_proba(self, x):
        """Return deterministic router probabilities for service tests."""
        self.calls += 1
        return np.repeat([[0.2, 0.8]], len(x), axis=0)


def _service(router):
    """Build an inference service fixture with mocked model artifacts."""
    return InferenceService(
        model=_Model([0.05, 0.95]),
        original_model=_Model([0.95, 0.05]),
        pipeline={
            "target_encoder": _TargetEncoder(),
            "encoders": {},
            "scaler_num": _IdentityScaler(),
            "final_scaler": _IdentityScaler(),
            "valid_cat_cols": [],
            "num_cols": ["f"],
        },
        final_features=["f"],
        artifact_dir="/tmp/test",
        domain_router={"model": router, "threshold": 0.6},
    )


def test_live_domain_uses_automatic_router():
    """Verify that live domain uses automatic router."""
    router = _Router()
    result = _service(router).predict([{"domain": "live", "f": 1.0}])[0]

    assert router.calls == 1
    assert result["route"] == "custom"
    assert result["predicted_label"] == "xss"
    assert result["router_confidence"] == 0.8


def test_original_domain_still_forces_base_expert():
    """Verify that original domain still forces base expert."""
    router = _Router()
    result = _service(router).predict([{"domain": "original", "f": 1.0}])[0]

    assert router.calls == 0
    assert result["route"] == "original"
    assert result["predicted_label"] == "normal"
    assert result["router_confidence"] == 1.0


def test_resnet_base_dos_ddos_class_is_alerting_label():
    """Verify that resnet base dos ddos class is alerting label."""
    service = InferenceService(
        model=_Model([0.05, 0.95]),
        pipeline={
            "target_encoder": _DosTargetEncoder(),
            "encoders": {},
            "scaler_num": _IdentityScaler(),
            "final_scaler": _IdentityScaler(),
            "valid_cat_cols": [],
            "num_cols": ["f"],
        },
        final_features=["f"],
        artifact_dir="/tmp/test",
    )

    result = service.predict([{"f": 1.0}])[0]

    assert service.class_names == ["normal", "ddos_dos"]
    assert result["predicted_label"] == "ddos_dos"
    assert "ddos_dos" in result["probabilities"]


def test_routed_experts_can_use_separate_pipelines_and_class_order():
    """Verify that routed experts can use separate pipelines and class order."""
    router = _Router()
    original_model = _RecordingModel([0.05, 0.95])
    service = InferenceService(
        model=_Model([0.05, 0.95]),
        original_model=original_model,
        original_pipeline={
            "target_encoder": _ReversedTargetEncoder(),
            "encoders": {},
            "scaler_num": _IdentityScaler(),
            "final_scaler": _IdentityScaler(),
            "valid_cat_cols": [],
            "num_cols": ["base_f"],
        },
        original_final_features=["base_f"],
        pipeline={
            "target_encoder": _TargetEncoder(),
            "encoders": {},
            "scaler_num": _IdentityScaler(),
            "final_scaler": _IdentityScaler(),
            "valid_cat_cols": [],
            "num_cols": ["custom_f"],
        },
        final_features=["custom_f"],
        artifact_dir="/tmp/test",
        domain_router={"model": router, "threshold": 0.6},
    )

    result = service.predict([{"domain": "original", "base_f": 7.0, "custom_f": 2.0}])[0]

    assert result["route"] == "original"
    assert result["predicted_label"] == "normal"
    assert original_model.last_x.tolist() == [[7.0]]
