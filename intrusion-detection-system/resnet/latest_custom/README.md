# Latest Custom Zeek

Canonical trainer:

```bash
python -u resnet/latest_custom/train.py ...
```

This runs the local `train_onefile.py` trainer and keeps latest custom Zeek
defaults in `config.py`.

Dataset builder:

```bash
python -u resnet/latest_custom/build_dataset.py ...
```

Random split:

```bash
python -u resnet/latest_custom/train.py \
  --csv /data/ton-iot-project/fresh_start/data/zeek_crossval.csv \
  --output-dir /data/ton-iot-project/fresh_start/artifacts/se_dwnet_zeek_crossval_full_random \
  --split random \
  --final-holdout-size 0.10 \
  --target-k 192 \
  --batch-size 1024 \
  --max-epochs 100
```

Best temporal split:

```bash
python -u resnet/latest_custom/train.py \
  --csv /data/ton-iot-project/fresh_start/data/zeek_crossval.csv \
  --output-dir /data/ton-iot-project/fresh_start/artifacts/se_dwnet_zeek_crossval_temporal_label \
  --split temporal \
  --source-group-mode label \
  --final-holdout-size 0.10 \
  --target-k 192 \
  --dropout 0.45 \
  --label-smoothing 0.02 \
  --batch-size 1024 \
  --max-epochs 100
```
