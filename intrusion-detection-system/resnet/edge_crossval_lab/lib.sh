#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${LAB_DIR}/config.env" ]]; then
  # shellcheck source=/dev/null
  source "${LAB_DIR}/config.env"
else
  # shellcheck source=/dev/null
  source "${LAB_DIR}/config.env.example"
fi

is_private_ip() {
  local ip="$1"
  [[ "$ip" == "localhost" ]] && return 0
  [[ "$ip" =~ ^127\. ]] && return 0
  [[ "$ip" =~ ^10\. ]] && return 0
  [[ "$ip" =~ ^192\.168\. ]] && return 0
  [[ "$ip" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]] && return 0
  [[ "$ip" =~ ^100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\. ]] && return 0
  return 1
}

require_lab_target() {
  local ip="$1"
  if [[ "${ALLOW_NON_PRIVATE:-0}" == "1" ]]; then
    return 0
  fi
  if ! is_private_ip "$ip"; then
    echo "Refusing non-private target IP: $ip" >&2
    echo "Set ALLOW_NON_PRIVATE=1 only for an explicitly authorized lab target." >&2
    exit 2
  fi
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 127
  fi
}

ensure_dirs() {
  mkdir -p "${EDGE_RAW_DIR}" "${EDGE_CSV_DIR}"
}

print_lab_config() {
  echo "IFACE=${IFACE}"
  echo "TARGET_IP=${TARGET_IP}"
  echo "TARGET_URL=${TARGET_URL}"
  echo "KALI_IP=${KALI_IP}"
  echo "EDGE_CROSSVAL_ROOT=${EDGE_CROSSVAL_ROOT}"
}
