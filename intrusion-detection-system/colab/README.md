# Colab Experiments

These scripts are for rerunning the custom Zeek dataset experiments in Google
Colab without relying on the old H200 server paths.

## Expected Drive Layout

```text
/content/drive/MyDrive/thesis_ids/
  data/
    zeek_crossval.csv
    edge_public_zeek_same_features_100k.csv        # optional domain-transfer target
  artifacts/
```

Copy large CSVs to the Colab VM before training if Drive I/O is slow:

```bash
mkdir -p /content/data
cp /content/drive/MyDrive/thesis_ids/data/zeek_crossval.csv /content/data/
```

Then pass `/content/data/zeek_crossval.csv` as `--csv` while keeping
`--output-dir` on Drive.

## Install

In a Colab notebook:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content/bi-lstm-ton-iot
!pip -q install -r colab/requirements_colab.txt
```

Colab usually already has TensorFlow. If it does not, install a TensorFlow
version compatible with the active Colab runtime.

## SE-DWNet Custom Zeek Training

```bash
!python -u colab/train_custom_sedwnet_colab.py \
  --csv /content/drive/MyDrive/thesis_ids/data/zeek_crossval.csv \
  --output-dir /content/drive/MyDrive/thesis_ids/artifacts/se_dwnet_zeek_crossval_colab \
  --split random \
  --final-holdout-size 0.05 \
  --final-holdout-mode random \
  --target-k 192 \
  --smote auto \
  --batch-size 1024 \
  --max-epochs 100
```

For the stricter drift test:

```bash
!python -u colab/train_custom_sedwnet_colab.py \
  --csv /content/drive/MyDrive/thesis_ids/data/zeek_crossval.csv \
  --output-dir /content/drive/MyDrive/thesis_ids/artifacts/se_dwnet_zeek_crossval_temporal_colab \
  --split temporal \
  --final-holdout-size 0.05 \
  --final-holdout-mode random \
  --target-k 192 \
  --smote auto \
  --batch-size 1024 \
  --max-epochs 100
```

## Classical Baselines

```bash
!python -u colab/train_custom_baselines_colab.py \
  --csv /content/drive/MyDrive/thesis_ids/data/zeek_crossval.csv \
  --output-dir /content/drive/MyDrive/thesis_ids/artifacts/custom_zeek_baselines_random \
  --split random \
  --final-holdout-size 0.05 \
  --final-holdout-mode random \
  --target-k 192 \
  --smote auto \
  --models rf,extratrees,histgb,sgd_logreg,linear_svm
```

If Colab free RAM is tight, start with:

```bash
!python -u colab/train_custom_baselines_colab.py \
  --csv /content/drive/MyDrive/thesis_ids/data/zeek_crossval.csv \
  --output-dir /content/drive/MyDrive/thesis_ids/artifacts/custom_zeek_baselines_fast \
  --models rf,extratrees,histgb,sgd_logreg \
  --n-estimators 100 \
  --max-train-rows-per-class 20000
```

## Domain Transfer: Custom -> Edge-IIoTset Zeek

This evaluates a custom-trained model on an Edge-IIoTset CSV that was extracted
with the same Zeek feature builder.

```bash
!python -u colab/domain_transfer_custom_to_edge_colab.py \
  --artifact-dir /content/drive/MyDrive/thesis_ids/artifacts/se_dwnet_zeek_crossval_colab \
  --target-csv /content/drive/MyDrive/thesis_ids/data/edge_public_zeek_same_features_100k.csv \
  --output-dir /content/drive/MyDrive/thesis_ids/artifacts/domain_transfer_custom_to_edge
```

To include baseline models in the same domain-transfer run:

```bash
!python -u colab/domain_transfer_custom_to_edge_colab.py \
  --artifact-dir /content/drive/MyDrive/thesis_ids/artifacts/se_dwnet_zeek_crossval_colab \
  --baseline-artifact-dir /content/drive/MyDrive/thesis_ids/artifacts/custom_zeek_baselines_random \
  --target-csv /content/drive/MyDrive/thesis_ids/data/edge_public_zeek_same_features_100k.csv \
  --output-dir /content/drive/MyDrive/thesis_ids/artifacts/domain_transfer_custom_to_edge
```

## Free vs Paid Colab

Start with free Colab. SE-DWNet should fit in a T4 session for the 360k-row
custom dataset. Classical tree baselines are CPU/RAM-bound; RandomForest and
ExtraTrees are the likely bottlenecks. Use `--max-train-rows-per-class 20000`
or fewer trees first. Move to a paid/high-RAM plan only if the runtime is killed
or memory is exhausted repeatedly.
