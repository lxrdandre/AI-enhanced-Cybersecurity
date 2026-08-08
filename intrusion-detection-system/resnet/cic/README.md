# CIC Public

Canonical trainer:

```bash
python -u resnet/cic/train.py ...
```

This runs the local `train_onefile.py` trainer. It reserves a spare validation
dataset with `--spare-validation-per-class` and automatically validates it
after training.

Public PCAP-to-Zeek builder:

```bash
python -u resnet/cic/build_zeek_dataset.py ...
```

Random split:

```bash
python -u resnet/cic/train.py \
  --csv /data/ton-iot-project/fresh_start/data/cic_public_6class.csv \
  --output-dir /data/ton-iot-project/fresh_start/artifacts/se_dwnet_cic_public_random \
  --split random \
  --spare-validation-per-class 1000 \
  --target-k 40 \
  --batch-size 1024 \
  --max-epochs 100
```

Temporal split:

```bash
python -u resnet/cic/train.py \
  --csv /data/ton-iot-project/fresh_start/data/cic_public_6class.csv \
  --output-dir /data/ton-iot-project/fresh_start/artifacts/se_dwnet_cic_public_temporal \
  --split temporal \
  --temporal-fallback error \
  --spare-validation-per-class 1000 \
  --target-k 40 \
  --batch-size 1024 \
  --max-epochs 100
```
