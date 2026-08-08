#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${LAB_DIR}/lib.sh"

SOURCE_LABEL="${1:-}"
TYPE_LABEL="${2:-}"
DURATION="${3:-60}"

if [[ -z "$SOURCE_LABEL" || -z "$TYPE_LABEL" ]]; then
  echo "Usage: $0 <source_label> <type_label> [duration_seconds]" >&2
  echo "Example: $0 ddos_tcp_syn dos_ddos 60" >&2
  exit 2
fi

require_lab_target "$TARGET_IP"
require_cmd tcpdump
ensure_dirs

PCAP="${EDGE_RAW_DIR}/${SOURCE_LABEL}.pcap"
META="${EDGE_RAW_DIR}/${SOURCE_LABEL}.meta.json"

echo "Capturing ${SOURCE_LABEL}/${TYPE_LABEL} for ${DURATION}s"
print_lab_config
echo "PCAP=${PCAP}"

sudo timeout "${DURATION}" tcpdump -i "${IFACE}" -w "${PCAP}" \
  "host ${TARGET_IP} or host ${KALI_IP}" || true

cat > "${META}" <<EOF
{
  "source_label": "${SOURCE_LABEL}",
  "type": "${TYPE_LABEL}",
  "duration_seconds": ${DURATION},
  "target_ip": "${TARGET_IP}",
  "kali_ip": "${KALI_IP}",
  "interface": "${IFACE}",
  "pcap": "${PCAP}"
}
EOF

echo "Saved ${PCAP}"
echo "Offline extraction command, for later:"
echo "python3 ${LAB_DIR}/pcap_to_edge_csv.py --pcap ${PCAP} --type ${TYPE_LABEL} --source-label ${SOURCE_LABEL} --output ${EDGE_CSV_DIR}/${SOURCE_LABEL}.csv"
