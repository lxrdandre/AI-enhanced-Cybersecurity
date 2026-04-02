#!/usr/bin/env bash
# Install and enable ClawdBot systemd services on the SVM.
# Usage:  sudo bash deploy/install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="/etc/systemd/system"

echo "==> Copying unit files to ${DEST}"
cp "${SCRIPT_DIR}/ids-api.service"        "${DEST}/ids-api.service"
cp "${SCRIPT_DIR}/clawdbot-agent.service" "${DEST}/clawdbot-agent.service"

echo "==> Reloading systemd"
systemctl daemon-reload

echo "==> Enabling services"
systemctl enable ids-api.service
systemctl enable clawdbot-agent.service

echo "==> Starting services"
systemctl start ids-api.service
# Wait for IDS API to be ready before starting the agent
sleep 3
systemctl start clawdbot-agent.service

echo "==> Status"
systemctl status ids-api.service --no-pager
systemctl status clawdbot-agent.service --no-pager

echo ""
echo "Done. Check logs with:"
echo "  journalctl -u ids-api -f"
echo "  journalctl -u clawdbot-agent -f"
