import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: List[str],
    report_path: str,
) -> str:
    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(report_path, "w") as f:
        f.write(report_str)
    return report_str


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: List[str],
    title: str,
    out_path: str,
    figsize=(12, 10),
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
