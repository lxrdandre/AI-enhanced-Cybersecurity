#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${LAB_DIR}/lib.sh"

ACTION="${1:-apply}"
NFT_TABLE="${NFT_TABLE:-edge_lab}"

# Restricted to Kali's source IP. The broad low-port ranges are intentional for
# scan realism: permitted-but-closed ports become "closed" instead of "filtered".
LAB_TCP_PORTS="${LAB_TCP_PORTS:-1-1024,1883,2000,2323,2404,3306,5432,6379,8000-8090,8443,9000,44818,${SSH_PORT:-22}}"
LAB_UDP_PORTS="${LAB_UDP_PORTS:-1-1024,1900,5353,5683,47808}"

require_lab_target "$KALI_IP"

find_nft() {
  if [[ -n "${NFT_BIN:-}" && -x "$NFT_BIN" ]]; then
    printf '%s\n' "$NFT_BIN"
    return 0
  fi
  if command -v nft >/dev/null 2>&1; then
    command -v nft
    return 0
  fi
  for candidate in /usr/sbin/nft /sbin/nft /usr/local/sbin/nft; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

NFT_BIN="$(find_nft || true)"
if [[ -z "$NFT_BIN" ]]; then
  cat >&2 <<'EOF'
Missing nft command.

On Debian/Ubuntu install it with:
  sudo apt-get update
  sudo apt-get install -y nftables

If it is installed in a non-standard location, run:
  NFT_BIN=/path/to/nft sudo -E bash allow_iot_lab_nft.sh apply
EOF
  exit 127
fi

normalize_services() {
  printf '%s\n' "$1" | tr ',' '\n' | awk '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
    {
      v = trim($0)
      if (v ~ /^[0-9]+-[0-9]+$/) {
        split(v, a, "-")
        if (a[1] >= 1 && a[1] <= 65535 && a[2] >= a[1] && a[2] <= 65535 && !seen[v]++) {
          printf "%s%s", sep, v
          sep = ", "
        }
      } else if (v ~ /^[0-9]+$/) {
        if (v >= 1 && v <= 65535 && !seen[v]++) {
          printf "%s%s", sep, v
          sep = ", "
        }
      }
    }
    END { print "" }
  '
}

TCP_PORTS="$(normalize_services "$LAB_TCP_PORTS")"
UDP_PORTS="$(normalize_services "$LAB_UDP_PORTS")"

apply_rules() {
  if [[ -z "$TCP_PORTS" || -z "$UDP_PORTS" ]]; then
    echo "Empty TCP/UDP port list after normalization." >&2
    exit 2
  fi

  sudo "$NFT_BIN" delete table inet "$NFT_TABLE" >/dev/null 2>&1 || true
  sudo "$NFT_BIN" -f - <<EOF
add table inet ${NFT_TABLE}
add set inet ${NFT_TABLE} lab_tcp_ports { type inet_service; flags interval; elements = { ${TCP_PORTS} } }
add set inet ${NFT_TABLE} lab_udp_ports { type inet_service; flags interval; elements = { ${UDP_PORTS} } }
add chain inet ${NFT_TABLE} input { type filter hook input priority -450; policy accept; }
add rule inet ${NFT_TABLE} input ip saddr ${KALI_IP} tcp dport @lab_tcp_ports counter accept comment "edge_crossval_lab tcp from kali"
add rule inet ${NFT_TABLE} input ip saddr ${KALI_IP} udp dport @lab_udp_ports counter accept comment "edge_crossval_lab udp from kali"
add rule inet ${NFT_TABLE} input ip saddr ${KALI_IP} ip protocol icmp counter accept comment "edge_crossval_lab icmp from kali"
EOF

  echo "Applied nft lab allow table: inet ${NFT_TABLE}"
  echo "Allowed source: ${KALI_IP}"
  echo "TCP ports: ${TCP_PORTS}"
  echo "UDP ports: ${UDP_PORTS}"
  echo
  echo "If Nmap still shows filtered, your existing firewall has a later drop chain."
  echo "In that case insert equivalent accept rules into the real input chain before its drop rule."
}

delete_rules() {
  sudo "$NFT_BIN" delete table inet "$NFT_TABLE" >/dev/null 2>&1 || true
  echo "Deleted nft table inet ${NFT_TABLE} if it existed."
}

save_rules() {
  if [[ ! -d /etc ]]; then
    echo "/etc not found; cannot save nftables config." >&2
    exit 1
  fi
  sudo "$NFT_BIN" list ruleset | sudo tee /etc/nftables.conf >/dev/null
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl enable nftables >/dev/null 2>&1 || true
  fi
  echo "Saved current nft ruleset to /etc/nftables.conf."
}

show_rules() {
  sudo "$NFT_BIN" list table inet "$NFT_TABLE"
}

case "$ACTION" in
  apply|add)
    apply_rules
    ;;
  delete|remove|flush)
    delete_rules
    ;;
  save)
    save_rules
    ;;
  show)
    show_rules
    ;;
  *)
    cat >&2 <<EOF
Usage: sudo bash $0 [apply|delete|show|save]

Environment overrides:
  KALI_IP=${KALI_IP}
  LAB_TCP_PORTS=${LAB_TCP_PORTS}
  LAB_UDP_PORTS=${LAB_UDP_PORTS}
EOF
    exit 2
    ;;
esac
