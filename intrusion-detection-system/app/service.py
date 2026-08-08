from __future__ import annotations

import hashlib
import os

import numpy as np

from app.preprocessing import transform_with_pipeline


class InferenceService:
    """Run IDS inference for the single classifier selected by artifact path."""

    def __init__(
        self,
        *,
        model,
        pipeline: dict,
        final_features: list[str],
        artifact_dir: str,
        calibration: dict | None = None,
        unknown_confidence_threshold: float = 0.45,
    ):
        """Store loaded model artifacts and derive supported class metadata."""
        self.model = model
        self.pipeline = pipeline
        self.final_features = final_features
        self.artifact_dir = artifact_dir
        self.temperature = calibration["temperature"] if calibration else 1.0
        target_encoder = pipeline.get("target_encoder")
        if target_encoder is None:
            raise RuntimeError("Pipeline does not include 'target_encoder'.")
        self.class_names = [self._canon_class(name) for name in target_encoder.classes_.tolist()]
        self.model_name = os.path.basename(artifact_dir)
        self.model_family = self._infer_model_family()
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
        """Return the canonical alert label for a model class name."""
        label = str(label)
        return "ddos_dos" if label in {"dos", "ddos", "dos_ddos", "ddos_dos"} else label

    def validate_records(self, records: list[dict]) -> None:
        """Validate raw records before converting them into a dataframe."""
        for idx, record in enumerate(records):
            missing = [field_name for field_name in self.required_fields if field_name not in record]
            if missing:
                raise ValueError(
                    f"records[{idx}] missing required fields: {', '.join(missing)}"
                )

    def predict(self, records: list[dict]) -> list[dict]:
        """Run inference and return normalized prediction dictionaries."""
        self.validate_records(records)
        x = transform_with_pipeline(records, pipeline=self.pipeline, final_features=self.final_features)
        probabilities = self._single_output(self.model.predict(x, verbose=0))

        if self.temperature != 1.0:
            log_probs = np.log(np.clip(probabilities, 1e-7, 1.0))
            scaled = log_probs / self.temperature
            probabilities = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)

        pred_indices = np.argmax(probabilities, axis=1)

        results = []
        for idx, prob_vector in zip(pred_indices, probabilities):
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
                }
            )
        return results

    def _single_output(self, raw) -> np.ndarray:
        """Normalize Keras output variants into a probability array."""
        if isinstance(raw, dict):
            output = self._first_present(raw, ("output", "predictions", "classifier", "custom_head", "original_head"))
            return np.asarray(output if output is not None else next(iter(raw.values())))
        if isinstance(raw, (list, tuple)):
            names = list(getattr(self.model, "output_names", []))
            by_name = dict(zip(names, raw))
            output = self._first_present(by_name, ("output", "predictions", "classifier", "custom_head", "original_head"))
            return np.asarray(output if output is not None else raw[0])
        return np.asarray(raw)

    @staticmethod
    def _first_present(values: dict, keys: tuple[str, ...]):
        """Return the first dataframe column present from a candidate list."""
        for key in keys:
            if key in values:
                return values[key]
        return None

    def _infer_model_family(self) -> str:
        """Infer the live feature family expected by the selected artifact."""
        features = set(self.final_features)
        artifact_name = os.path.basename(self.artifact_dir).lower()
        if any(feature.startswith("ctx_") for feature in features) or {"id_orig_p", "history"} <= features:
            return "zeek_crossval"
        if {"arp_hw_size", "mqtt_hdrflags", "tcp_flags"} & features:
            return "edge_iiotset"
        if {"flow_byts_s", "fwd_iat_tot", "bwd_header_len"} & features:
            return "cic_public"
        if {"conn_state", "dns_query", "src_ip_bytes", "dst_ip_bytes"} & features:
            return "ton_iot"
        if "zeek" in artifact_name:
            return "zeek_crossval"
        if "edge" in artifact_name:
            return "edge_iiotset"
        if "cic" in artifact_name:
            return "cic_public"
        if "ton" in artifact_name:
            return "ton_iot"
        return "generic"

    def metadata(self) -> dict:
        """Return model, feature, and class metadata."""
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "artifact_dir": self.artifact_dir,
            "class_names": self.class_names,
            "feature_count": self.feature_count,
            "input_dim": self.input_dim,
            "required_fields": self.required_fields,
            "available_fields": sorted(self.available_fields),
            "feature_signature_sha256": self.feature_signature_sha256,
            "unknown_confidence_threshold": self.unknown_confidence_threshold,
            "routing_enabled": False,
            "route_fields": [],
        }
