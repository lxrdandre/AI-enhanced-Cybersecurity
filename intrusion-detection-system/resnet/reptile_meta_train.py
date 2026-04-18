"""
Reptile meta-training on the TON-IoT 7-class base dataset.

Produces meta-initialized weights that can rapidly adapt to new attack classes
with only a handful of labelled samples (few-shot).

Usage
-----
    python -m resnet.reptile_meta_train            # defaults
    python -m resnet.reptile_meta_train --iters 5000 --n-way 5

Output
------
    artifacts/resnet_reptile/
        resnet_reptile_meta.keras          – meta-initialised model
        preprocessing_pipeline.pkl         – copied from base
        final_features.txt                 – copied from base
        reptile_meta_config.json           – full training config + history summary
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from app.preprocessing import SafeLabelEncoder, transform_with_pipeline


# ── Defaults ──────────────────────────────────────────────────────────────────
SEED = 42

# Reptile hyper-parameters
META_ITERS = 3000
META_BATCH_TASKS = 4
N_WAY = 7
K_SHOT = 32
Q_QUERY = 32
INNER_STEPS = 5
INNER_LR = 3e-4
INNER_CLIPNORM = 1.0
META_LR_START = 0.02
META_LR_END = 0.002
VAL_EVERY = 100
VAL_EPISODES = 20
INNER_FORWARD_TRAINING = False
FREEZE_BATCH_NORM = True

# Base artifact pair candidates (searched in order)
ARTIFACT_PAIRS = [
    (
        os.path.join("artifacts", "resnet_base", "resnet_model.keras"),
        os.path.join("artifacts", "resnet_base", "preprocessing_pipeline.pkl"),
    ),
    (
        os.path.join("artifacts", "resnet_transfer_7class", "resnet_transfer_model_7class.keras"),
        os.path.join("artifacts", "resnet_transfer_7class", "preprocessing_pipeline.pkl"),
    ),
]

OUTPUT_DIR_REL = os.path.join("artifacts", "resnet_reptile")


# ── Path helpers ──────────────────────────────────────────────────────────────
def _detect_project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, ".."))


def _resolve_artifact_pair(project_root: str) -> tuple[str, str]:
    """Find the first existing model+pipeline pair."""
    for model_rel, pipeline_rel in ARTIFACT_PAIRS:
        model_path = os.path.join(project_root, model_rel)
        pipeline_path = os.path.join(project_root, pipeline_rel)
        if os.path.exists(model_path) and os.path.exists(pipeline_path):
            return model_path, pipeline_path
    tried = [f"  - {m} + {p}" for m, p in ARTIFACT_PAIRS]
    raise FileNotFoundError(
        "No matching model+pipeline pair found. Tried:\n" + "\n".join(tried)
    )


def _load_final_features(pipeline_dir: str, pipeline: dict) -> list[str]:
    txt_path = os.path.join(pipeline_dir, "final_features.txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            feats = [line.strip() for line in f if line.strip()]
        if feats:
            return feats
    feats = pipeline.get("features")
    if not feats:
        raise RuntimeError("Cannot determine final features.")
    return [str(feat).strip() for feat in feats if str(feat).strip()]


def _find_dataset(project_root: str) -> str:
    candidates = [
        os.path.join(project_root, "data", "train_test_network.csv"),
        os.path.join(project_root, "data", "Network_dataset_capped.csv"),
        os.path.join(project_root, "data", "network_dataset_capped.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Training CSV not found. Tried:\n" + "\n".join(f"  - {p}" for p in candidates)
    )


# ── Data preparation ─────────────────────────────────────────────────────────
def _prepare_data(
    project_root: str,
    pipeline: dict,
    final_features: list[str],
) -> dict:
    """Load the base TON-IoT dataset, preprocess, and return train/val splits."""
    csv_path = _find_dataset(project_root)
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False, dtype=str, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df.drop(columns=["ts", "date", "time"], errors="ignore", inplace=True)

    # Identify label column
    label_col = None
    for c in ("type", "label", "Label"):
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise RuntimeError("No label column found.")

    if label_col != "type":
        df.rename(columns={label_col: "type"}, inplace=True)
    df.drop(columns=["label"], errors="ignore", inplace=True)

    # Drop rare classes
    labels_norm = df["type"].astype(str).str.strip().str.lower()
    drop_labels = {"mitm", "ransomware"}
    df = df.loc[~labels_norm.isin(drop_labels)].copy()

    # Merge dos+ddos → dos_ddos (half-sample each)
    labels_norm = df["type"].astype(str).str.strip().str.lower()
    dos_df = df.loc[labels_norm == "dos"]
    ddos_df = df.loc[labels_norm == "ddos"]
    other_df = df.loc[~labels_norm.isin({"dos", "ddos"})]

    dos_half = dos_df.sample(n=max(1, len(dos_df) // 2), random_state=SEED) if len(dos_df) > 0 else dos_df
    ddos_half = ddos_df.sample(n=max(1, len(ddos_df) // 2), random_state=SEED) if len(ddos_df) > 0 else ddos_df
    dos_half = dos_half.copy()
    ddos_half = ddos_half.copy()
    dos_half["type"] = "dos_ddos"
    ddos_half["type"] = "dos_ddos"

    df = pd.concat([other_df, dos_half, ddos_half], ignore_index=True)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Merged dos({len(dos_half)})+ddos({len(ddos_half)}) → dos_ddos")

    # Drop IP columns
    ip_cols = [c for c in ["src_ip", "dst_ip", "srcip", "dstip"] if c in df.columns]
    if ip_cols:
        df.drop(columns=ip_cols, inplace=True)

    # Filter to classes the base pipeline's target_encoder knows
    target_encoder = pipeline.get("target_encoder")
    if target_encoder is None:
        raise RuntimeError("Pipeline missing target_encoder.")
    known_classes = set(target_encoder.classes_.tolist())
    df = df[df["type"].astype(str).isin(known_classes)].copy()
    if df.empty:
        raise RuntimeError("Dataset empty after filtering to known classes.")

    y_encoded = target_encoder.transform(df["type"].astype(str).values)
    x_df = df.drop(columns=["type"]).copy()

    # Random stratified split 80/20
    x_train_df, x_val_df, y_train, y_val = train_test_split(
        x_df, y_encoded, test_size=0.2, stratify=y_encoded, random_state=SEED,
    )
    print(f"Random split — Train: {len(x_train_df)}  Val: {len(x_val_df)}")

    del x_df, df
    gc.collect()

    x_train = transform_with_pipeline(
        x_train_df.to_dict(orient="records"),
        pipeline=pipeline, final_features=final_features,
    )
    x_val = transform_with_pipeline(
        x_val_df.to_dict(orient="records"),
        pipeline=pipeline, final_features=final_features,
    )

    del x_train_df, x_val_df
    gc.collect()

    return {
        "x_train": x_train.astype(np.float32),
        "y_train": np.asarray(y_train, dtype=np.int32),
        "x_val": x_val.astype(np.float32),
        "y_val": np.asarray(y_val, dtype=np.int32),
        "num_classes": len(target_encoder.classes_),
        "class_names": target_encoder.classes_.tolist(),
        "csv_path": _find_dataset(project_root),
    }


# ── Episodic sampler ─────────────────────────────────────────────────────────
class EpisodicSampler:
    """Samples N-way K-shot + Q-query episodes from a labelled dataset."""

    def __init__(self, x: np.ndarray, y: np.ndarray, seed: int = 42):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int32)
        self.rng = np.random.default_rng(seed)
        self.class_ids = np.unique(self.y)
        self.class_to_indices = {
            int(cid): np.where(self.y == cid)[0] for cid in self.class_ids
        }

    def sample_episode(
        self, n_way: int, k_shot: int, q_query: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if n_way > len(self.class_ids):
            raise ValueError(f"n_way={n_way} > available classes={len(self.class_ids)}")

        chosen = self.rng.choice(self.class_ids, size=n_way, replace=False)
        sx, sy, qx, qy = [], [], [], []

        for cls in chosen:
            indices = self.class_to_indices[int(cls)]
            needed = k_shot + q_query
            pick = self.rng.choice(indices, size=needed, replace=len(indices) < needed)
            sx.append(self.x[pick[:k_shot]])
            sy.append(self.y[pick[:k_shot]])
            qx.append(self.x[pick[k_shot:]])
            qy.append(self.y[pick[k_shot:]])

        sx, sy = np.concatenate(sx), np.concatenate(sy)
        qx, qy = np.concatenate(qx), np.concatenate(qy)

        sp, qp = self.rng.permutation(len(sy)), self.rng.permutation(len(qy))
        return sx[sp], sy[sp], qx[qp], qy[qp]


# ── Inner-loop adaptation ────────────────────────────────────────────────────
def _inner_adapt(
    model: Model,
    x_support: np.ndarray,
    y_support: np.ndarray,
    *,
    inner_steps: int,
    inner_lr: float,
    clipnorm: float,
    loss_fn,
    training: bool,
) -> None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=inner_lr, clipnorm=clipnorm)
    x_tf = tf.convert_to_tensor(x_support, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y_support, dtype=tf.int32)

    for _ in range(inner_steps):
        with tf.GradientTape() as tape:
            logits = model(x_tf, training=training)
            loss = tf.reduce_mean(loss_fn(y_tf, logits))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


def _episodic_val_accuracy(
    model: Model,
    sampler: EpisodicSampler,
    *,
    episodes: int,
    n_way: int,
    k_shot: int,
    q_query: int,
    inner_steps: int,
    inner_lr: float,
    clipnorm: float,
    loss_fn,
    training: bool,
) -> float:
    """Evaluate: adapt on support, measure accuracy on query. Returns mean accuracy."""
    base_weights = model.get_weights()
    accs = []

    for _ in range(episodes):
        sx, sy, qx, qy = sampler.sample_episode(n_way, k_shot, q_query)
        model.set_weights(base_weights)
        _inner_adapt(
            model, sx, sy,
            inner_steps=inner_steps, inner_lr=inner_lr,
            clipnorm=clipnorm, loss_fn=loss_fn, training=training,
        )
        probs = model.predict(qx, batch_size=max(256, len(qy)), verbose=0)
        accs.append(float(np.mean(np.argmax(probs, axis=1) == qy)))

    model.set_weights(base_weights)
    return float(np.mean(accs)) if accs else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reptile meta-training on TON-IoT base dataset")
    parser.add_argument("--iters", type=int, default=META_ITERS, help="Meta-training iterations")
    parser.add_argument("--meta-batch", type=int, default=META_BATCH_TASKS, help="Tasks per meta-step")
    parser.add_argument("--n-way", type=int, default=N_WAY, help="Classes per episode")
    parser.add_argument("--k-shot", type=int, default=K_SHOT, help="Support samples per class")
    parser.add_argument("--q-query", type=int, default=Q_QUERY, help="Query samples per class")
    parser.add_argument("--inner-steps", type=int, default=INNER_STEPS, help="Inner-loop gradient steps")
    parser.add_argument("--inner-lr", type=float, default=INNER_LR, help="Inner-loop learning rate")
    parser.add_argument("--meta-lr-start", type=float, default=META_LR_START)
    parser.add_argument("--meta-lr-end", type=float, default=META_LR_END)
    parser.add_argument("--val-every", type=int, default=VAL_EVERY, help="Validate every N iters")
    parser.add_argument("--val-episodes", type=int, default=VAL_EPISODES)
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args(argv)

    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    # GPU setup
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    project_root = _detect_project_root()
    out_dir = args.output_dir or os.path.join(project_root, OUTPUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Reptile Meta-Training — TON-IoT SE-DWNet")
    print("=" * 60)
    print(f"Project root:  {project_root}")
    print(f"Output:        {out_dir}")
    print(f"GPUs:          {len(tf.config.list_physical_devices('GPU'))}")

    # Resolve base artifacts
    base_model_path, base_pipeline_path = _resolve_artifact_pair(project_root)
    print(f"Base model:    {base_model_path}")
    print(f"Base pipeline: {base_pipeline_path}")

    with open(base_pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    final_features = _load_final_features(os.path.dirname(base_pipeline_path), pipeline)
    print(f"Features:      {len(final_features)}")

    # Prepare data
    data = _prepare_data(project_root, pipeline, final_features)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    num_classes = data["num_classes"]
    class_names = data["class_names"]
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}")

    if args.n_way > num_classes:
        print(f"Warning: n_way={args.n_way} > classes={num_classes}, clamping to {num_classes}")
        args.n_way = num_classes

    train_sampler = EpisodicSampler(x_train, y_train, seed=SEED)
    val_sampler = EpisodicSampler(x_val, y_val, seed=SEED + 1)

    # Load model
    model = tf.keras.models.load_model(base_model_path, compile=False)
    assert int(model.output_shape[-1]) == num_classes, (
        f"Model outputs {model.output_shape[-1]} classes, dataset has {num_classes}"
    )
    assert int(model.input_shape[-1]) == x_train.shape[1], (
        f"Model input dim {model.input_shape[-1]} != feature dim {x_train.shape[1]}"
    )

    # Freeze BatchNorm for inner-loop stability
    if FREEZE_BATCH_NORM:
        bn_count = 0
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
                bn_count += 1
        print(f"Frozen BatchNorm layers: {bn_count}")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # Initial validation
    meta_weights = model.get_weights()
    initial_val_acc = _episodic_val_accuracy(
        model, val_sampler,
        episodes=args.val_episodes, n_way=args.n_way,
        k_shot=args.k_shot, q_query=args.q_query,
        inner_steps=args.inner_steps, inner_lr=args.inner_lr,
        clipnorm=INNER_CLIPNORM, loss_fn=loss_fn,
        training=INNER_FORWARD_TRAINING,
    )
    print(f"Initial episodic val accuracy: {initial_val_acc:.4f}")

    best_val_acc = initial_val_acc
    best_weights = list(meta_weights)
    history: list[dict] = []

    # ── Reptile meta-training loop ────────────────────────────────────────────
    print(f"\nStarting Reptile ({args.iters} iters, {args.meta_batch} tasks/step)...")
    t0 = time.time()

    for meta_iter in range(1, args.iters + 1):
        progress = (meta_iter - 1) / max(args.iters - 1, 1)
        meta_lr = (1.0 - progress) * args.meta_lr_start + progress * args.meta_lr_end

        deltas = [np.zeros_like(w) for w in meta_weights]
        support_accs = []

        for _ in range(args.meta_batch):
            model.set_weights(meta_weights)
            sx, sy, _, _ = train_sampler.sample_episode(args.n_way, args.k_shot, args.q_query)

            _inner_adapt(
                model, sx, sy,
                inner_steps=args.inner_steps, inner_lr=args.inner_lr,
                clipnorm=INNER_CLIPNORM, loss_fn=loss_fn,
                training=INNER_FORWARD_TRAINING,
            )

            probs = model.predict(sx, batch_size=max(256, len(sy)), verbose=0)
            support_accs.append(float(np.mean(np.argmax(probs, axis=1) == sy)))

            task_weights = model.get_weights()
            for i in range(len(deltas)):
                deltas[i] += task_weights[i] - meta_weights[i]

        # Reptile outer-step
        meta_weights = [
            meta_weights[i] + (meta_lr / args.meta_batch) * deltas[i]
            for i in range(len(meta_weights))
        ]
        model.set_weights(meta_weights)

        row = {
            "iter": meta_iter,
            "meta_lr": round(meta_lr, 6),
            "support_acc": round(float(np.mean(support_accs)), 4),
        }

        # Periodic validation
        if meta_iter % args.val_every == 0 or meta_iter == 1 or meta_iter == args.iters:
            val_acc = _episodic_val_accuracy(
                model, val_sampler,
                episodes=args.val_episodes, n_way=args.n_way,
                k_shot=args.k_shot, q_query=args.q_query,
                inner_steps=args.inner_steps, inner_lr=args.inner_lr,
                clipnorm=INNER_CLIPNORM, loss_fn=loss_fn,
                training=INNER_FORWARD_TRAINING,
            )
            row["val_acc"] = round(val_acc, 4)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = model.get_weights()

            elapsed = time.time() - t0
            eta = elapsed / meta_iter * (args.iters - meta_iter)
            print(
                f"[{meta_iter:>{len(str(args.iters))}}/{args.iters}] "
                f"meta_lr={meta_lr:.5f}  support_acc={row['support_acc']:.4f}  "
                f"val_acc={val_acc:.4f}  best={best_val_acc:.4f}  "
                f"ETA={eta/60:.1f}min"
            )
        elif meta_iter % 50 == 0:
            print(
                f"[{meta_iter:>{len(str(args.iters))}}/{args.iters}] "
                f"meta_lr={meta_lr:.5f}  support_acc={row['support_acc']:.4f}"
            )

        history.append(row)

    # ── Save artifacts ────────────────────────────────────────────────────────
    model.set_weights(best_weights)

    model_path = os.path.join(out_dir, "resnet_reptile_meta.keras")
    model.save(model_path)
    print(f"\nSaved meta-model: {model_path}")

    # Copy pipeline + features into output dir
    pipeline_dir = os.path.dirname(base_pipeline_path)
    for fname in ("preprocessing_pipeline.pkl", "final_features.txt", "calibration.pkl"):
        src = os.path.join(pipeline_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))
            print(f"Copied {fname} → {out_dir}")

    # Save config
    config = {
        "algorithm": "reptile",
        "seed": SEED,
        "meta_iters": args.iters,
        "meta_batch_tasks": args.meta_batch,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "q_query": args.q_query,
        "inner_steps": args.inner_steps,
        "inner_lr": args.inner_lr,
        "inner_clipnorm": INNER_CLIPNORM,
        "inner_forward_training": INNER_FORWARD_TRAINING,
        "freeze_batch_norm": FREEZE_BATCH_NORM,
        "meta_lr_start": args.meta_lr_start,
        "meta_lr_end": args.meta_lr_end,
        "base_model": base_model_path,
        "base_pipeline": base_pipeline_path,
        "dataset": data["csv_path"],
        "classes": class_names,
        "num_classes": num_classes,
        "initial_val_acc": round(initial_val_acc, 4),
        "best_val_acc": round(best_val_acc, 4),
        "total_time_minutes": round((time.time() - t0) / 60, 2),
    }
    config_path = os.path.join(out_dir, "reptile_meta_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save history
    import csv as csv_mod
    history_path = os.path.join(out_dir, "reptile_training_history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["iter", "meta_lr", "support_acc", "val_acc"])
        writer.writeheader()
        writer.writerows(history)

    print(f"\nDONE. Best episodic val accuracy: {best_val_acc:.4f}")
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
