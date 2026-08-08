#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib.sh
source "${LAB_DIR}/lib.sh"

DURATION="${1:-60}"
MQTT_HOST="${MQTT_HOST:-${TARGET_IP}}"
MQTT_PORT="${MQTT_PORT:-1883}"

require_lab_target "$MQTT_HOST"
require_cmd mosquitto_pub

echo "Generating normal/mqtt sensor traffic for ${DURATION}s to ${MQTT_HOST}:${MQTT_PORT}"
end=$((SECONDS + DURATION))
while (( SECONDS < end )); do
  mosquitto_pub -h "$MQTT_HOST" -p "$MQTT_PORT" -t "factory/line1/temperature" -m "$((20 + RANDOM % 8)).$((RANDOM % 10))" || true
  mosquitto_pub -h "$MQTT_HOST" -p "$MQTT_PORT" -t "factory/line1/humidity" -m "$((35 + RANDOM % 30))" || true
  mosquitto_pub -h "$MQTT_HOST" -p "$MQTT_PORT" -t "factory/line1/water_level" -m "$((RANDOM % 100))" || true
  sleep 0.5
done
