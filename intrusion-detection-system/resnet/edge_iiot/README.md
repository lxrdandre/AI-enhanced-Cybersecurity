# Edge-IIoTset

Canonical trainer:

```bash
python -u resnet/edge_iiot/train.py ...
```

This runs the local `train_onefile.py` trainer and keeps Edge-IIoTset defaults
in `config.py`.

Dataset builder:

```bash
python -u resnet/edge_iiot/build_dataset.py ...
```

Random split:

```bash
python -u resnet/edge_iiot/train.py \
  --csv /data/datasets/edge_iiotset/processed/edge_iiotset_6class_cap100k.csv \
  --output-dir /data/ton-iot-project/fresh_start/artifacts/se_dwnet_edge_iiotset_random \
  --split random \
  --target-k 40 \
  --batch-size 1024 \
  --max-epochs 100
```

Temporal split:

```bash
python -u resnet/edge_iiot/train.py \
  --csv /data/datasets/edge_iiotset/processed/edge_iiotset_6class_cap100k_temporal.csv \
  --output-dir /data/ton-iot-project/fresh_start/artifacts/se_dwnet_edge_iiotset_temporal \
  --split temporal \
  --temporal-fallback error \
  --target-k 40 \
  --batch-size 1024 \
  --max-epochs 100
```
