# TON-IoT

Canonical trainer:

```bash
python -u resnet/ton_iot/train.py ...
```

This runs the local `train_onefile.py` trainer and keeps TON-IoT defaults in
`config.py`.

Validation dataset helper:

```bash
python -u resnet/ton_iot/build_dataset.py ...
```

Random split:

```bash
python -u resnet/ton_iot/train.py \
  --csv /data/ton-iot-project/fresh_start/data/train_test_network.csv \
  --output-dir /data/ton-iot-project/fresh_start/artifacts/se_dwnet_toniot_random \
  --dataset-name toniot_random \
  --split random \
  --target-k 25 \
  --batch-size 1024 \
  --max-epochs 100
```

Temporal split:

```bash
python -u resnet/ton_iot/train.py \
  --csv /data/ton-iot-project/fresh_start/data/train_test_network.csv \
  --output-dir /data/ton-iot-project/fresh_start/artifacts/se_dwnet_toniot_temporal \
  --dataset-name toniot_temporal \
  --split temporal \
  --temporal-fallback error \
  --target-k 25 \
  --batch-size 1024 \
  --max-epochs 100
```
