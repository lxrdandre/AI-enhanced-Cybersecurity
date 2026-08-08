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


def _service():
    """Build an inference service fixture with mocked model artifacts."""
    return InferenceService(
        model=_Model([0.05, 0.95]),
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
    )


def test_legacy_domain_field_does_not_route():
    """Verify that legacy route hints do not switch classifiers."""
    result = _service().predict([{"domain": "original", "f": 1.0}])[0]

    assert result["predicted_label"] == "xss"
    assert "route" not in result
    assert "router_confidence" not in result


def test_metadata_reports_single_classifier():
    """Verify that model metadata reports routing disabled."""
    metadata = _service().metadata()

    assert metadata["routing_enabled"] is False
    assert metadata["route_fields"] == []
    assert metadata["model_family"] == "generic"


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


def test_multi_output_model_uses_classifier_named_output():
    """Verify that multi-output artifacts are treated as one selected classifier."""
    class _MultiOutputModel(_Model):
        """Represent a legacy multi-output model."""
        output_names = ["original_head", "custom_head"]

        def predict(self, x, verbose=0):
            """Return two output heads; classifier-named output is used."""
            return [
                np.repeat(np.asarray([[0.90, 0.10]], dtype=np.float32), len(x), axis=0),
                np.repeat(np.asarray([[0.05, 0.95]], dtype=np.float32), len(x), axis=0),
            ]

    service = InferenceService(
        model=_MultiOutputModel([0.05, 0.95]),
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
    )

    result = service.predict([{"f": 2.0}])[0]

    assert result["predicted_label"] == "xss"
