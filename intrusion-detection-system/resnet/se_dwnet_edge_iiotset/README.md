# SE-DWNet Edge-IIoTset Training

This folder provides the modular entry point for the Edge-IIoTset 6-class
SE-DWNet classifier.

The original single-file trainer is still available at:

```bash
resnet/resnet_edge_iiotset.py
```

Run the modular entry point:

```bash
python -m resnet.se_dwnet_edge_iiotset \
  --csv /data/datasets/edge_iiotset/processed/edge_iiotset_6class_cap100k_source.csv \
  --target-k 40 \
  --batch-size 1024 \
  --max-epochs 100 \
  --split random \
  --final-holdout-size 0.05 \
  --val-size 0.15 \
  --test-size 0.15 \
  --no-dedupe
```

Default output:

```text
artifacts/se_dwnet_edge_iiotset_random_holdout/
```

The wrapper keeps legacy files such as `resnet_model.keras`, but also creates
the active deployment alias:

```text
se_dwnet_model.keras
```
