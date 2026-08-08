"""Artifact naming helpers for SE-DWNet Edge-IIoTset training."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import LEGACY_MODEL_FILENAME, MODEL_FILENAME


TEXT_REPLACEMENTS = {
    "ResNet Edge-IIoTset": "SE-DWNet Edge-IIoTset",
    "resnet_edge_iiotset_random_holdout": "se_dwnet_edge_iiotset_random_holdout",
    "resnet_edge_iiotset": "se_dwnet_edge_iiotset",
    LEGACY_MODEL_FILENAME: MODEL_FILENAME,
}


def _rewrite_text_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def _copy_if_present(src: Path, dst: Path) -> None:
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def normalize_artifact_names(output_dir: str | Path) -> dict[str, str]:
    """Add SE-DWNet artifact aliases while preserving legacy files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _copy_if_present(out / LEGACY_MODEL_FILENAME, out / MODEL_FILENAME)
    _copy_if_present(
        out / "resnet_edge_iiotset_confusion_matrix.png",
        out / "se_dwnet_edge_iiotset_confusion_matrix.png",
    )
    _copy_if_present(
        out / "resnet_edge_iiotset_final_holdout_confusion_matrix.png",
        out / "se_dwnet_edge_iiotset_final_holdout_confusion_matrix.png",
    )

    for path in out.glob("*.txt"):
        _rewrite_text_file(path)
    for path in out.glob("*.json"):
        _rewrite_text_file(path)

    metadata_path = out / "training_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["model_family"] = "SE-DWNet"
        metadata["model_path"] = str(out / MODEL_FILENAME)
        metadata["artifact_dir"] = str(out)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "artifact_dir": str(out),
        "model": str(out / MODEL_FILENAME),
        "legacy_model": str(out / LEGACY_MODEL_FILENAME),
        "pipeline": str(out / "preprocessing_pipeline.pkl"),
        "features": str(out / "final_features.txt"),
    }
