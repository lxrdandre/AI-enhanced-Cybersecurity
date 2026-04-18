from __future__ import annotations

import numpy as np

from app.service import InferenceService


class _IdentityScaler:
    def transform(self, values):
        return values


class _TargetEncoder:
    classes_ = np.array(["normal", "xss"])


class _Model:
    input_shape = (None, 1)

    def __init__(self, probs):
        self.probs = np.asarray([probs], dtype=np.float32)

    def predict(self, x, verbose=0):
        return np.repeat(self.probs, len(x), axis=0)


class _Router:
    classes_ = np.array([0, 1])

    def __init__(self):
        self.calls = 0

    def predict_proba(self, x):
        self.calls += 1
        return np.repeat([[0.2, 0.8]], len(x), axis=0)


def _service(router):
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
    router = _Router()
    result = _service(router).predict([{"domain": "live", "f": 1.0}])[0]

    assert router.calls == 1
    assert result["route"] == "custom"
    assert result["predicted_label"] == "xss"
    assert result["router_confidence"] == 0.8


def test_original_domain_still_forces_base_expert():
    router = _Router()
    result = _service(router).predict([{"domain": "original", "f": 1.0}])[0]

    assert router.calls == 0
    assert result["route"] == "original"
    assert result["predicted_label"] == "normal"
    assert result["router_confidence"] == 1.0
