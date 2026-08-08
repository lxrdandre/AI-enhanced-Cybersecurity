# Two-VM Quickstart: Kali Generates, Debian Captures

Use this when you want realistic attacker/victim traffic:

- Kali VM runs the attack generators.
- Debian victim/IDS VM runs `tcpdump`, extracts CSV rows, counts classes, and
  builds the final dataset.

No GPU is needed on either VM.

## 1. Copy The Same Folder To Both VMs

From your laptop:

```bash
rsync -avz edge_crossval_lab/ kali@192.168.56.10:~/edge_crossval_lab/
rsync -avz edge_crossval_lab/ ids@192.168.56.20:~/edge_crossval_lab/
```

Adjust users/IPs to your lab.

## 2. Kali VM Setup

```bash
cd ~/edge_crossval_lab
cp config.env.example config.env
nano config.env
```

Kali `config.env` should point to the Debian victim:

```bash
export TARGET_IP="192.168.56.20"
export HTTP_PORT="8080"
export TARGET_URL="http://192.168.56.20:8080"
export KALI_IP="192.168.56.10"
export SSH_PORT="22"
export CANONICAL_SSH_PORT="22"
export C2_PORT="8090"
export C2_URL="http://192.168.56.20:8090"
```

Install generator tools:

```bash
sudo apt-get update
sudo apt-get install -y curl dnsutils iputils-ping nmap hping3 hydra apache2-utils python3
```

Optional:

```bash
sudo apt-get install -y nikto mosquitto-clients
```

Check the worker:

```bash
source config.env
bash kali_attack_worker.sh --check
```

Important: the Debian controller calls this over SSH. For overnight unattended
runs, use either `root` SSH to Kali or passwordless sudo for the Kali user,
because `hping3` and SYN/OS scans need raw socket privileges.

## 3. SSH From Debian To Kali

On Debian victim:

```bash
ssh-keygen -t ed25519
ssh-copy-id kali@192.168.56.10
ssh kali@192.168.56.10 'cd ~/edge_crossval_lab && bash kali_attack_worker.sh --check'
```

If you use a different user, set it in Debian `config.env`.

## 4. Debian Victim Setup

```bash
cd ~/edge_crossval_lab
cp config.env.example config.env
nano config.env
```

Debian `config.env`:

```bash
export IFACE="ens18"
export TARGET_IP="192.168.56.20"
export HTTP_PORT="8080"
export TARGET_URL="http://192.168.56.20:8080"
export KALI_IP="192.168.56.10"
export SSH_PORT="22"
export CANONICAL_SSH_PORT="22"

export KALI_SSH_USER="kali"
export KALI_SSH_PORT="22"
export KALI_LAB_DIR="~/edge_crossval_lab"

export MAX_ROUNDS="14"
export START_LOCAL_C2="1"
export C2_PORT="8090"
export C2_URL="http://192.168.56.20:8090"
```

If the Debian victim does not already expose IoT-like services, start the
benign fake service runner before capture. It opens web admin, camera, MQTT,
Modbus, CoAP, SSDP/UPnP, SNMP-like UDP, and other IoT/OT-looking ports:

```bash
cd ~/edge_crossval_lab
sudo bash allow_iot_lab_nft.sh apply
sudo python3 iot_lab_services.py
```

To keep those nft rules after reboot:

```bash
sudo bash allow_iot_lab_nft.sh save
```

From Kali, verify the HTTP and scan surface before capture:

```bash
curl -I http://192.168.56.20:8080
nmap -sS -sV -O -p 21,23,80,81,102,443,502,554,631,1883,2323,2404,4840,5357,8080,8081,9000,44818 192.168.56.20
```

Install capture/build tools:

```bash
sudo apt-get update
sudo apt-get install -y tcpdump zeek tshark python3 python3-pip openssh-client
python3 -m pip install --user pandas
```

## 5. Run The Robust Two-Terminal Schedule

This mode avoids Debian SSHing into Kali. Start one schedule on Debian and one
schedule on Kali with the same `SCHEDULE_START_EPOCH`.

Pick a start time two minutes in the future:

```bash
START=$(( $(date +%s) + 120 ))
echo "$START"
```

On Debian:

```bash
cd ~/edge_crossval_lab
source config.env
export SCHEDULE_START_EPOCH=PASTE_START_VALUE_HERE
bash run_debian_capture_schedule.sh
```

On Kali:

```bash
cd ~/edge_crossval_lab
source config.env
export SCHEDULE_START_EPOCH=PASTE_START_VALUE_HERE
bash run_kali_attack_schedule.sh
```

This is the recommended mode if the SSH-driven controller hangs.

## 6. Older SSH-Driven Runner

```bash
cd ~/edge_crossval_lab
source config.env
bash run_debian_capture_builder.sh
```

The Debian runner:

1. starts local capture with `tcpdump`
2. SSHes to Kali and runs `kali_attack_worker.sh <source_label>`
3. writes PCAPs under `data/edge_crossval/raw`
4. repeats for `MAX_ROUNDS`

It does not extract CSVs or build the dataset during capture.

Outputs on Debian:

```text
data/edge_crossval/raw/*.pcap
data/edge_crossval/logs/*.log
```

Check progress:

```bash
tail -f data/edge_crossval/logs/*.log
ls -lh data/edge_crossval/raw
```

## 7. Extract And Build Later On Debian

After capture completes:

```bash
bash build_dataset_from_pcaps.sh
```

The default builder uses Zeek and writes:

```text
data/edge_crossval/zeek_crossval.csv
data/edge_crossval/zeek_crossval_report.json
data/edge_crossval/zeek_crossval_counts.json
```

Use this Zeek dataset for the new retraining script. The old TShark extractor
is still available with `PCAP_PARSER=tshark`, but it is no longer the
recommended path.
