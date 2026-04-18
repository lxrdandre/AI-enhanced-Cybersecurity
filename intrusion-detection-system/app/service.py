from __future__ import annotations

import hashlib
import os

import numpy as np

from app.preprocessing import transform_with_pipeline


class InferenceService:
    ROUTE_FIELDS = ("domain", "_domain", "source_domain", "_source", "source")
    CUSTOM_ROUTE_VALUES = {"custom", "tpot", "honeypot", "transfer", "custom_like", "custom-like"}
    ORIGINAL_ROUTE_VALUES = {"original", "base", "toniot", "ton_iot", "ton-iot", "holdout", "network"}

    def __init__(
        self,
        *,
        model,
        original_model=None,
        pipeline: dict,
        final_features: list[str],
        artifact_dir: str,
        calibration: dict | None = None,
        domain_router: dict | None = None,
        unknown_confidence_threshold: float = 0.45,
    ):
        self.model = model
        self.original_model = original_model
        self.pipeline = pipeline
        self.final_features = final_features
        self.artifact_dir = artifact_dir
        self.domain_router = domain_router
        self.temperature = calibration["temperature"] if calibration else 1.0
        target_encoder = pipeline.get("target_encoder")
        if target_encoder is None:
            raise RuntimeError("Pipeline does not include 'target_encoder'.")
        self.class_names = [self._canon_class(name) for name in target_encoder.classes_.tolist()]
        self.model_name = os.path.basename(artifact_dir)
        valid_cat_cols = [str(col) for col in pipeline.get("valid_cat_cols", [])]
        num_cols = [str(col) for col in pipeline.get("num_cols", [])]
        self.available_fields = set(valid_cat_cols + num_cols + self.final_features)
        self.required_fields = [
            field_name
            for field_name in ("duration", "src_bytes", "dst_bytes", "proto")
            if field_name in self.available_fields
        ]
        self.feature_count = len(self.final_features)
        self.input_dim = int(self.model.input_shape[-1])
        joined_features = "\n".join(self.final_features).encode("utf-8")
        self.feature_signature_sha256 = hashlib.sha256(joined_features).hexdigest()
        self.unknown_confidence_threshold = unknown_confidence_threshold

    @staticmethod
    def _canon_class(label: object) -> str:
        label = str(label)
        return "dos_ddos" if label in {"dos", "ddos", "ddos_dos"} else label

    def validate_records(self, records: list[dict]) -> None:
        for idx, record in enumerate(records):
            missing = [field_name for field_name in self.required_fields if field_name not in record]
            if missing:
                raise ValueError(
                    f"records[{idx}] missing required fields: {', '.join(missing)}"
                )

    def predict(self, records: list[dict]) -> list[dict]:
        self.validate_records(records)
        x = transform_with_pipeline(records, pipeline=self.pipeline, final_features=self.final_features)
        original_probs, custom_probs = self._predict_experts(x)
        if custom_probs is None:
            probabilities = original_probs
            routes = ["single"] * len(records)
            route_scores = [None] * len(records)
        else:
            routes, route_scores = self._routes(records, x)
            probabilities = np.vstack([
                custom_probs[idx] if route == "custom" else original_probs[idx]
                for idx, route in enumerate(routes)
            ])

        if self.temperature != 1.0:
            log_probs = np.log(np.clip(probabilities, 1e-7, 1.0))
            scaled = log_probs / self.temperature
            probabilities = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)

        pred_indices = np.argmax(probabilities, axis=1)

        results = []
        for idx, prob_vector, route, route_score in zip(pred_indices, probabilities, routes, route_scores):
            idx_int = int(idx)
            max_confidence = float(prob_vector[idx_int])
            prob_map = {label: float(prob) for label, prob in zip(self.class_names, prob_vector)}

            if max_confidence < self.unknown_confidence_threshold:
                predicted_label = "unknown"
            else:
                predicted_label = self.class_names[idx_int]

            results.append(
                {
                    "predicted_index": idx_int,
                    "predicted_label": predicted_label,
                    "confidence": max_confidence,
                    "probabilities": prob_map,
                    "route": route,
                    "router_confidence": route_score,
                }
            )
        return results

    def _predict_experts(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.original_model is not None:
            original = np.asarray(self.original_model.predict(x, verbose=0))
            custom = self._single_output(self.model.predict(x, verbose=0))
            return original, custom

        return self._predict_heads(x)

    def _predict_heads(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        raw = self.model.predict(x, verbose=0)
        if isinstance(raw, dict):
            original = self._first_present(raw, ("original_head", "original"))
            custom = self._first_present(raw, ("custom_head", "custom", "transfer_head"))
            if original is not None and custom is not None:
                return np.asarray(original), np.asarray(custom)
            return np.asarray(next(iter(raw.values()))), None

        if isinstance(raw, (list, tuple)):
            names = list(getattr(self.model, "output_names", []))
            by_name = dict(zip(names, raw))
            original = self._first_present(by_name, ("original_head", "original"))
            custom = self._first_present(by_name, ("custom_head", "custom", "transfer_head"))
            if original is not None and custom is not None:
                return np.asarray(original), np.asarray(custom)
            if len(raw) >= 2:
                return np.asarray(raw[0]), np.asarray(raw[1])
            return np.asarray(raw[0]), None

        return np.asarray(raw), None

    def _single_output(self, raw) -> np.ndarray:
        if isinstance(raw, dict):
            custom = self._first_present(raw, ("custom_head", "custom", "transfer_head"))
            return np.asarray(custom if custom is not None else next(iter(raw.values())))
        if isinstance(raw, (list, tuple)):
            names = list(getattr(self.model, "output_names", []))
            by_name = dict(zip(names, raw))
            custom = self._first_present(by_name, ("custom_head", "custom", "transfer_head"))
            return np.asarray(custom if custom is not None else raw[-1])
        return np.asarray(raw)

    @staticmethod
    def _first_present(values: dict, keys: tuple[str, ...]):
        for key in keys:
            if key in values:
                return values[key]
        return None

    def _routes(self, records: list[dict], x: np.ndarray) -> tuple[list[str], list[float | None]]:
        routes: list[str | None] = [self._manual_route(record) for record in records]
        scores: list[float | None] = [1.0 if route is not None else None for route in routes]
        unresolved = [idx for idx, route in enumerate(routes) if route is None]

        if unresolved and self.domain_router is not None:
            router = self.domain_router.get("model", self.domain_router)
            threshold = float(self.domain_router.get("threshold", 0.60))
            proba = router.predict_proba(x[unresolved])
            classes = list(getattr(router, "classes_", [0, 1]))
            custom_idx = classes.index(1) if 1 in classes else classes.index("custom")
            for row_idx, sample_idx in enumerate(unresolved):
                p_custom = float(proba[row_idx, custom_idx])
                route = "custom" if p_custom >= threshold else "original"
                routes[sample_idx] = route
                scores[sample_idx] = p_custom if route == "custom" else 1.0 - p_custom

        return [route or "original" for route in routes], scores

    def _manual_route(self, record: dict) -> str | None:
        for field in self.ROUTE_FIELDS:
            value = record.get(field)
            if value in (None, ""):
                continue
            normalized = str(value).strip().lower().replace(" ", "_")
            if normalized in self.CUSTOM_ROUTE_VALUES:
                return "custom"
            if normalized in self.ORIGINAL_ROUTE_VALUES:
                return "original"
        return None

    def metadata(self) -> dict:
        return {
            "model_name": self.model_name,
            "artifact_dir": self.artifact_dir,
            "class_names": self.class_names,
            "feature_count": self.feature_count,
            "input_dim": self.input_dim,
            "required_fields": self.required_fields,
            "feature_signature_sha256": self.feature_signature_sha256,
            "unknown_confidence_threshold": self.unknown_confidence_threshold,
            "routing_enabled": self.original_model is not None or self.domain_router is not None or len(getattr(self.model, "output_names", [])) > 1,
            "route_fields": list(self.ROUTE_FIELDS),
        }
