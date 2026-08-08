# Edge-Like Cross-Validation Capture Plan

Use these commands only inside your authorized lab network. The scripts refuse
non-private targets unless you set `ALLOW_NON_PRIVATE=1`.

The default lab values from the old notes are:

```bash
export IFACE=ens18
export TARGET_IP=192.168.56.20
export HTTP_PORT=8080
export TARGET_URL=http://192.168.56.20:8080
export KALI_IP=192.168.56.10
export SSH_PORT=22
export CANONICAL_SSH_PORT=22
export C2_PORT=8090
export C2_URL=http://192.168.56.20:8090
```

## Setup

On the capture host:

```bash
cd /opt/ton-iot-ips
cp resnet/edge_crossval_lab/config.env.example resnet/edge_crossval_lab/config.env
nano resnet/edge_crossval_lab/config.env
source resnet/edge_crossval_lab/config.env
```

## Capture Pairs

For each row below, start the capture command in one terminal, then run the
generator command from Kali or the host that should produce that traffic.

| Source label | Type | Capture command | Generator command |
| --- | --- | --- | --- |
| `normal_http_dns_icmp` | `normal` | `bash resnet/edge_crossval_lab/capture_source.sh normal_http_dns_icmp normal 180` | `bash resnet/edge_crossval_lab/attacks/normal_mix.sh mixed 180` |
| `normal_mqtt` | `normal` | `bash resnet/edge_crossval_lab/capture_source.sh normal_mqtt normal 120` | `bash resnet/edge_crossval_lab/attacks/mqtt_normal.sh 120` |
| `ddos_tcp_syn` | `dos_ddos` | `bash resnet/edge_crossval_lab/capture_source.sh ddos_tcp_syn dos_ddos 45` | `bash resnet/edge_crossval_lab/attacks/dos_ddos.sh tcp_syn 45 "$HTTP_PORT"` |
| `ddos_udp` | `dos_ddos` | `bash resnet/edge_crossval_lab/capture_source.sh ddos_udp dos_ddos 45` | `bash resnet/edge_crossval_lab/attacks/dos_ddos.sh udp 45 "$HTTP_PORT"` |
| `ddos_icmp` | `dos_ddos` | `bash resnet/edge_crossval_lab/capture_source.sh ddos_icmp dos_ddos 45` | `bash resnet/edge_crossval_lab/attacks/dos_ddos.sh icmp 45` |
| `ddos_http` | `dos_ddos` | `bash resnet/edge_crossval_lab/capture_source.sh ddos_http dos_ddos 90` | `bash resnet/edge_crossval_lab/attacks/dos_ddos.sh http 90 "$HTTP_PORT"` |
| `port_scanning` | `scanning` | `bash resnet/edge_crossval_lab/capture_source.sh port_scanning scanning 90` | `bash resnet/edge_crossval_lab/attacks/scanning.sh port` |
| `os_fingerprinting` | `scanning` | `bash resnet/edge_crossval_lab/capture_source.sh os_fingerprinting scanning 90` | `bash resnet/edge_crossval_lab/attacks/scanning.sh os` |
| `vulnerability_scanner` | `scanning` | `bash resnet/edge_crossval_lab/capture_source.sh vulnerability_scanner scanning 180` | `bash resnet/edge_crossval_lab/attacks/scanning.sh vuln` |
| `password_ssh` | `password` | `bash resnet/edge_crossval_lab/capture_source.sh password_ssh password 180` | `SSH_USER=test bash resnet/edge_crossval_lab/attacks/password.sh ssh 180` |
| `password_http` | `password` | `bash resnet/edge_crossval_lab/capture_source.sh password_http password 120` | `bash resnet/edge_crossval_lab/attacks/password.sh http 120` |
| `sql_injection` | `injection` | `bash resnet/edge_crossval_lab/capture_source.sh sql_injection injection 120` | `bash resnet/edge_crossval_lab/attacks/injection.sh sql 2500` |
| `xss` | `injection` | `bash resnet/edge_crossval_lab/capture_source.sh xss injection 120` | `bash resnet/edge_crossval_lab/attacks/injection.sh xss 2500` |
| `uploading` | `injection` | `bash resnet/edge_crossval_lab/capture_source.sh uploading injection 120` | `bash resnet/edge_crossval_lab/attacks/injection.sh uploading 2500` |
| `backdoor_http_c2` | `backdoor` | `bash resnet/edge_crossval_lab/capture_source.sh backdoor_http_c2 backdoor 180` | `python3 resnet/edge_crossval_lab/attacks/backdoor_beacon_client.py --url "$C2_URL" --duration 180` |

For `backdoor_http_c2`, first run this on the target/C2 side:

```bash
python3 resnet/edge_crossval_lab/attacks/backdoor_c2_server.py --host 0.0.0.0 --port "$C2_PORT"
```

## Extract And Build

After captures finish:

```bash
python3 resnet/edge_crossval_lab/extract_all_pcaps.py \
  --raw-dir data/edge_crossval/raw \
  --csv-dir data/edge_crossval/csv \
  --overwrite

python3 resnet/edge_crossval_lab/build_edge_crossval_dataset.py \
  --input-dir data/edge_crossval/csv \
  --output-csv data/edge_crossval/edge_like_crossval.csv \
  --report-json data/edge_crossval/edge_like_crossval_report.json \
  --distribution balanced \
  --cap-per-major-class 60000 \
  --backdoor-cap 60000 \
  --quota-mode even
```

Expected target shape with that command is at most:

```text
backdoor: 60000
dos_ddos: 60000
injection: 60000
normal: 60000
password: 60000
scanning: 60000
```

If one source is too small, the builder keeps everything available and reports
the shortfall in `edge_like_crossval_report.json`.

## Validate

```bash
python3 -u resnet/validate_edge_iiotset_dataset.py \
  --csv data/edge_crossval/edge_like_crossval.csv \
  --model-dir artifacts/se_dwnet_edge_iiotset_random_holdout \
  --dataset-name edge_like_crossval \
  --label-col type \
  --output-dir artifacts/se_dwnet_edge_iiotset_random_holdout/edge_like_crossval_validation
```
