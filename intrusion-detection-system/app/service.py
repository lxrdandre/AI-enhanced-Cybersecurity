from __future__ import annotations

import hashlib
import os

import numpy as np

from app.preprocessing import transform_with_pipeline


class InferenceService:
    def __init__(self, *, model, pipeline: dict, final_features: list[str], artifact_dir: str, calibration: dict | None = None, unknown_confidence_threshold: float = 0.45):
        self.model = model
        self.pipeline = pipeline
        self.final_features = final_features
        self.artifact_dir = artifact_dir
        self.temperature = calibration["temperature"] if calibration else 1.0
        target_encoder = pipeline.get("target_encoder")
        if target_encoder is None:
            raise RuntimeError("Pipeline does not include 'target_encoder'.")
        self.class_names = [str(name) for name in target_encoder.classes_.tolist()]
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
        probabilities = self.model.predict(x, verbose=0)

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
        }
