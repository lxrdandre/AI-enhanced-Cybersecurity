from __future__ import annotations

import hashlib
import json
import os
import time
from uuid import uuid4


def _records_hash(records: list[dict]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLogger:
    def __init__(self, *, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_analyze(
        self,
        *,
        model_name: str,
        records: list[dict],
        predictions: list[dict],
        triage: list[dict],
        llm_enabled: bool,
        llm_error: str | None,
    ) -> str:
        audit_id = uuid4().hex
        row = {
            "audit_id": audit_id,
            "timestamp": int(time.time()),
            "model_name": model_name,
            "record_count": len(records),
            "records_hash_sha256": _records_hash(records),
            "predictions": predictions,
            "triage": triage,
            "llm_enabled": llm_enabled,
            "llm_error": llm_error,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return audit_id
