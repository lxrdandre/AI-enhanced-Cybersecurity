# Edge-IIoTset-Style Cross-Validation Lab

This folder contains lab-only scripts for creating a diverse six-class
cross-validation or retraining dataset from your own cyber range traffic.

The scripts use defaults found in the old simulation notes:

```text
TARGET_IP=192.168.56.20
KALI_IP=192.168.56.10
IFACE=ens18
```

Override them with environment variables or edit `config.env.example`.

## Goal

Generate traffic sources similar to the Edge-IIoTset source families:

| Final class | Source labels generated here |
| --- | --- |
| `normal` | `normal_http`, `normal_dns`, `normal_icmp`, optional `normal_mqtt` |
| `dos_ddos` | `ddos_tcp_syn`, `ddos_udp`, `ddos_icmp`, `ddos_http` |
| `scanning` | `port_scanning`, `os_fingerprinting`, `vulnerability_scanner` |
| `password` | `password_ssh`, optional `password_http` |
| `injection` | `sql_injection`, `xss`, `uploading` |
| `backdoor` | `backdoor_http_c2` |

For retraining, the default builder uses Zeek, not TShark. Zeek produces
flow/protocol records that are more stable for live IDS work than sparse
packet-level TShark columns.

## Basic Workflow

On the capture/target side:

```bash
cd /opt/ton-iot-ips
cp resnet/edge_crossval_lab/config.env.example resnet/edge_crossval_lab/config.env
source resnet/edge_crossval_lab/config.env
```

For the easiest overnight PCAP-only run, use:

```bash
bash resnet/edge_crossval_lab/run_overnight_edge_crossval.sh
```

The standalone server instructions are in `SERVER_QUICKSTART.md`.
For the recommended two-VM setup, where Kali generates traffic and Debian
captures PCAPs before later extraction/building, use `TWO_VM_QUICKSTART.md`.
The most robust two-VM mode is `run_debian_capture_schedule.sh` on Debian plus
`run_kali_attack_schedule.sh` on Kali with the same `SCHEDULE_START_EPOCH`.

Before capture, the Debian victim should expose some benign IoT-looking
services so scans and HTTP floods do not hit only closed ports:

```bash
cd ~/edge_crossval_lab
sudo bash allow_iot_lab_nft.sh apply
sudo python3 iot_lab_services.py
```

Capture one traffic source while you run the matching attack/traffic generator:

```bash
bash resnet/edge_crossval_lab/capture_source.sh ddos_tcp_syn dos_ddos 60
```

In another terminal, run the matching generator from Kali or the right host:

```bash
bash resnet/edge_crossval_lab/attacks/dos_ddos.sh tcp_syn
```

After each PCAP is captured, extract Edge-style CSV:

```bash
python3 resnet/edge_crossval_lab/pcap_to_edge_csv.py \
  --pcap data/edge_crossval/raw/ddos_tcp_syn.pcap \
  --type dos_ddos \
  --source-label ddos_tcp_syn \
  --output data/edge_crossval/csv/ddos_tcp_syn.csv
```

Convert the captured PCAPs into a Zeek-flow dataset:

```bash
bash resnet/edge_crossval_lab/build_dataset_from_pcaps.sh
```

For a full command sequence, use `CAPTURE_PLAN.md`.

Validate against the active SE-DWNet Edge classifier:

```bash
python3 -u resnet/validate_edge_iiotset_dataset.py \
  --csv data/edge_crossval/edge_like_crossval.csv \
  --model-dir artifacts/se_dwnet_edge_iiotset_random_holdout \
  --dataset-name edge_like_crossval \
  --label-col type \
  --output-dir artifacts/se_dwnet_edge_iiotset_random_holdout/edge_like_crossval_validation
```

## Distribution Advice

For an Edge-like validation dataset, do not generate only one subtype per final
class. Try to keep merged classes subtype-balanced:

- `dos_ddos`: 25% TCP SYN, 25% UDP, 25% ICMP, 25% HTTP flood
- `scanning`: port scan + OS fingerprint + vuln scan
- `injection`: SQLi + XSS + upload/path traversal
- `normal`: HTTP + DNS + ICMP + optional MQTT/Modbus-like traffic

This makes the custom dataset closer to how the Edge-IIoTset model was trained.
