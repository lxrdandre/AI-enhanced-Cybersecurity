# Server Quickstart

This folder is standalone. Copy it to the machine where you will run the lab
traffic, edit `config.env`, and start the overnight capture runner. For the
better two-VM setup with Kali generating and Debian capturing, use
`TWO_VM_QUICKSTART.md`.

No GPU is needed. The runner uses CPU/network tools only.

## 1. Install Tools

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tcpdump zeek tshark curl dnsutils iputils-ping nmap hping3 hydra apache2-utils python3 python3-pip
python3 -m pip install --user pandas
```

Optional:

```bash
sudo apt-get install -y nikto mosquitto-clients
```

## 2. Configure

```bash
cd edge_crossval_lab
cp config.env.example config.env
nano config.env
```

Important values:

```bash
export IFACE="ens18"
export TARGET_IP="192.168.56.20"
export HTTP_PORT="8080"
export TARGET_URL="http://192.168.56.20:8080"
export KALI_IP="192.168.56.10"
export SSH_PORT="22"
export CANONICAL_SSH_PORT="22"
export MAX_ROUNDS="14"
```

If you want the script to start the benign C2 server locally for backdoor-like
traffic:

```bash
export START_LOCAL_C2="1"
export C2_PORT="8090"
export C2_URL="http://127.0.0.1:8090"
```

Otherwise, start `attacks/backdoor_c2_server.py` on the target and point
`C2_URL` to it.

For richer Nmap and normal HTTP results, run the benign IoT/OT fake services
on the victim in a separate terminal. First allow the lab traffic from Kali:

```bash
sudo bash allow_iot_lab_nft.sh apply
sudo python3 iot_lab_services.py
```

This opens common lab-only IoT ports such as HTTP admin, camera/RTSP, Modbus,
MQTT, CoAP, SSDP/UPnP, SNMP-like UDP, OPC UA, and EtherNet/IP.

## 3. Run Overnight

```bash
source config.env
bash run_overnight_edge_crossval.sh
```

The runner now captures PCAPs only. It stops after `MAX_ROUNDS` and writes:

```text
data/edge_crossval/raw/*.pcap
data/edge_crossval/logs/*.log
```

## 4. Extract And Build Later

After the capture run finishes:

```bash
bash build_dataset_from_pcaps.sh
```

This uses Zeek by default and writes:

```text
data/edge_crossval/zeek_crossval.csv
data/edge_crossval/zeek_crossval_report.json
data/edge_crossval/zeek_crossval_counts.json
```

To force the older TShark/Edge-style packet-field extractor:

```bash
PCAP_PARSER=tshark bash build_dataset_from_pcaps.sh
```

## 5. Check Outputs

```bash
python3 count_edge_csv_rows.py --input-dir data/edge_crossval/csv --json
cat data/edge_crossval/edge_like_crossval_report.json
```

## Notes

- `MAX_ROUNDS=14` controls how many capture cycles are run.
- `TARGET_PER_CLASS=60000` is used later by the final builder, not by capture;
  values above 60000 are clamped by `build_dataset_from_pcaps.sh`.
- DDoS will usually exceed 60k quickly; the builder caps it.
- Scanning and backdoor may need more rounds because they generate fewer rows.
- If a service is missing on the target, the related generator still creates
  failed connection traffic, but that may not look as close to Edge-IIoTset as
  successful HTTP/SSH/MQTT exchanges.
