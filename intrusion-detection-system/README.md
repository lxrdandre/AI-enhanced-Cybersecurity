Video of current functionalities - https://youtu.be/PqMEY2Pppp0

# TON IoT IPS - Deployable Inference Service + ClawdBot SOC Agent

Network intrusion detection system built on SE-DWNet/ResNet with a FastAPI inference API, tiered local LLM triage (Ollama), autonomous SOC response agent (ClawdBot via OpenClaw), and Telegram alerting.

**Hardware:** NVIDIA H200 (141 GB HBM3e), 40 GB RAM, 1 TB disk

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Step 1 - Python environment & IDS API](#step-1--python-environment--ids-api)
4. [Step 2 - Ollama (local LLM)](#step-2--ollama-local-llm)
5. [Step 3 - OpenClaw + ClawdBot](#step-3--openclaw--clawdbot)
6. [Step 4 - Telegram bot](#step-4--telegram-bot)
7. [Step 5 - ClawdBot capture agent](#step-5--clawdbot-capture-agent)
8. [Step 6 - Start everything](#step-6--start-everything)
9. [Step 7 - Live Dashboard](#step-7--live-dashboard)
10. [Systemd deployment](#systemd-deployment)
11. [Restarting services](#restarting-services)
12. [Environment variables reference](#environment-variables-reference)
13. [API endpoints](#api-endpoints)
14. [Tiered LLM escalation](#tiered-llm-escalation)
15. [Telegram alert format](#telegram-alert-format)
16. [Running tests](#running-tests)
17. [LLM model recommendation](#llm-model-recommendation)

---

## Architecture

```
 LAN traffic (scapy live capture)
          |
          v
   +--------------+
   |  ClawdBot     |  <- capture agent (python -m clawdbot)
   |  Agent        |     sniffs packets -> aggregates flows
   +------+-------+
          | POST /analyze (batch of flow records)
          v
   +--------------+
   |  IDS API      |  <- SE-DWNet / ResNet classifier
   |  /analyze     |     (this repo, port 8000)
   +------+-------+
          | predictions + triage
          v
   +--------------+
   |  Triage       |  <- MITRE ATT&CK labeling via Ollama
   |  (Ollama)     |
   +--+-------+---+
      |       |
      v       v
  Tier-1 LLM  Telegram Bot
  + Tier-2     SOC alerts

  Tier-1: mistral-small:24b   (fast, every alert)
  Tier-2: llama3.1:70b        (escalation only)
```

---

## Prerequisites

| Component | Version / Notes |
|-----------|----------------|
| Python 3.10+ | With venv |
| NVIDIA GPU + CUDA | H200 or compatible |
| Ollama | Local LLM server |
| OpenClaw | ClawdBot orchestrator |
| Telegram account | For SOC alerts |

---

## Step 1 - Python environment & IDS API

```bash
# Clone and enter the repo
cd /path/to/fresh_start

# Create virtual environment
python -m venv venv_h200
source venv_h200/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set project root
export TON_IOT_PROJECT_ROOT=$(pwd)
export TON_IOT_ARTIFACT_DIR=$(pwd)/artifacts/resnet_transfer_7class

# Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Verify
curl http://127.0.0.1:8000/health
```

---

## Step 2 - Ollama (local LLM)

### Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pull models

```bash
# Tier-1: fast model for every alert (~26 GB VRAM)
ollama pull mistral-small:24b

# Tier-2: escalation model for ambiguous/critical cases (~75 GB VRAM)
ollama pull llama3.1:70b-instruct-q8_0
```

### Start Ollama

```bash
# Start as a service (auto-starts on boot)
sudo systemctl enable ollama
sudo systemctl start ollama

# Or start manually
ollama serve &
```

### Verify

```bash
# Check models are loaded
ollama list

# Test connectivity
curl http://localhost:11434/api/tags

# Quick test inference
ollama run mistral-small:24b "Respond with only: OK"
```

### Set triage backend to Ollama

```bash
export TON_IOT_TRIAGE_BACKEND=ollama
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL_TIER1=mistral-small:24b
export OLLAMA_MODEL_TIER2=llama3.1:70b-instruct-q8_0
export OLLAMA_ESCALATION_CONFIDENCE=0.75
```

---

## Step 3 - OpenClaw + ClawdBot

### Install OpenClaw

Follow the [OpenClaw installation guide](https://openclaw.dev) for your system.

### Configure OpenClaw to use local Ollama

The OpenClaw CLI validates provider config as a whole block - individual field sets will fail validation. Use this Python script to patch the config correctly:

```bash
cat << 'PATCH_EOF' > patch_claw.py
import json

conf_path = "/home/adrian/.openclaw/openclaw.json"

with open(conf_path, 'r') as f:
    data = json.load(f)

# Add Ollama auth profile
data['auth']['profiles']['ollama:default'] = {
    "provider": "ollama",
    "mode": "api_key"
}

# Add Ollama provider with models
# Uses /v1 for OpenAI-compatible API layer
data['models']['providers']['ollama'] = {
    "baseUrl": "http://127.0.0.1:11434/v1",
    "api": "openai-completions",
    "apiKey": "ollama-local",
    "models": [
        {
            "id": "mistral-small:24b",
            "name": "Local Mistral Small 24B",
            "reasoning": False,
            "input": ["text"],
            "contextWindow": 32768,
            "maxTokens": 8192
        },
        {
            "id": "llama3.1:70b-instruct-q8_0",
            "name": "Local Llama 3.1 70B",
            "reasoning": False,
            "input": ["text"],
            "contextWindow": 131072,
            "maxTokens": 8192
        }
    ]
}

# Set primary agent model + fallback
data['agents']['defaults']['model']['primary'] = "ollama/mistral-small:24b"
data['agents']['defaults']['model']['fallbacks'] = ["ollama/llama3.1:70b-instruct-q8_0"]

with open(conf_path, 'w') as f:
    json.dump(data, f, indent=2)

print("OpenClaw config updated for local Ollama.")
PATCH_EOF

python3 patch_claw.py
```

> **Why a script instead of `openclaw config set`?** OpenClaw validates the entire provider block on each set command. Setting fields individually (baseUrl, apiKey, models) fails because the incomplete block doesn't pass validation. The script writes the full block atomically.

### Start OpenClaw gateway

```bash
openclaw gateway
```

---

## Step 4 - Telegram bot

### Create the bot

1. Open Telegram -> message **@BotFather** -> `/newbot`
2. Save the **bot token** (e.g. `7123456789:AAH...`)
3. Create a private group/channel for SOC alerts -> add the bot as admin
4. Get the **chat ID**:

```bash
# Send any message in the group first, then:
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates" | python3 -m json.tool
# Look for "chat": {"id": -100xxxxxxxxxx}
```

### Set Telegram env vars

```bash
export TELEGRAM_BOT_TOKEN="7123456789:AAHxxxxxx"
export TELEGRAM_CHAT_ID="-100xxxxxxxxxx"
```

---

## Step 5 - ClawdBot capture agent

The capture agent sniffs live LAN traffic with scapy, aggregates packets into flow records, POSTs them to the IDS API `/analyze` endpoint, and sends Telegram alerts for detected attacks.

### Run manually

```bash
source venv_h200/bin/activate

export CLAWDBOT_INTERFACE=eth0          # network interface to sniff
export CLAWDBOT_API_URL=http://127.0.0.1:8000
export CLAWDBOT_HARVEST_INTERVAL=10      # seconds between harvest cycles
export CLAWDBOT_SEVERITY_THRESHOLD=medium
export TELEGRAM_BOT_TOKEN="7123456789:AAHxxxxxx"
export TELEGRAM_CHAT_ID="-100xxxxxxxxxx"

# Requires root or CAP_NET_RAW for packet capture
sudo -E python -m clawdbot
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAWDBOT_INTERFACE` | **(required)** | Network interface (e.g. `eth0`, `ens18`) |
| `CLAWDBOT_BPF_FILTER` | `ip` | BPF filter for scapy (e.g. `tcp port 80`) |
| `CLAWDBOT_API_URL` | `http://127.0.0.1:8000` | IDS API base URL |
| `CLAWDBOT_API_TIMEOUT` | `30` | API request timeout (seconds) |
| `CLAWDBOT_HARVEST_INTERVAL` | `10` | Seconds between flow harvests |
| `CLAWDBOT_SEVERITY_THRESHOLD` | `medium` | Min severity to send Telegram alerts |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Step 6 - Start everything

Start all services in this order:

```bash
# 1. Ollama (if not already running as a systemd service)
ollama serve &

# 2. IDS API
source venv_h200/bin/activate
export TON_IOT_PROJECT_ROOT=$(pwd)
export TON_IOT_ARTIFACT_DIR=$(pwd)/artifacts/resnet_transfer_7class
export TON_IOT_TRIAGE_BACKEND=ollama
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 3. OpenClaw gateway (ClawdBot)
openclaw gateway &

# 4. ClawdBot capture agent (needs root for raw sockets)
export CLAWDBOT_INTERFACE=eth0
export TELEGRAM_BOT_TOKEN="your-token"
export TELEGRAM_CHAT_ID="your-chat-id"
sudo -E python -m clawdbot &

# 5. Verify all services
curl http://127.0.0.1:8000/health          # IDS API
curl http://127.0.0.1:11434/api/tags       # Ollama
```

---

## Step 7 - Live Dashboard

The Flask dashboard reads ClawdBot event logs and IDS audit logs, then refreshes SOC metrics in the browser every few seconds. It loads Chart.js and Google Fonts from public CDNs for the live charts and typography.

```bash
source venv_h200/bin/activate
export TON_IOT_PROJECT_ROOT=$(pwd)
export TON_IOT_DASHBOARD_HOST=0.0.0.0
export TON_IOT_DASHBOARD_PORT=5000
python -m dashboard.app
```

Open:

```text
http://SERVER_IP:5000
```

Dashboard data sources:

| Source | Default path | Used for |
|--------|--------------|----------|
| ClawdBot attacks | `logs/attacks.jsonl` | Latest incidents, severity, source/target/port rankings |
| ClawdBot actions | `logs/actions.jsonl` | Agent/firewall/system events |
| IDS audit | `artifacts/audit/analyze_events.jsonl` | Analyzed record counts, route mix, unknown rate, LLM errors |
| IDS API | `http://127.0.0.1:8000` | Online status and model metadata |

The Incident Stream marks whether a detection was sent to Telegram. Use the `Open` button to view MITRE ATT&CK mapping, model route/confidence, flow context, firewall/reputation status, and response/remediation actions.

---

## Systemd deployment

For production, use the systemd unit files in `deploy/`:

```bash
# Install and start all services (run on the SVM as root)
sudo bash deploy/install.sh
```

This installs three services:

| Service | Description | User |
|---------|-------------|------|
| `ids-api` | Uvicorn IDS API on port 8000 | `adrian` |
| `clawdbot-agent` | Capture agent (needs `CAP_NET_RAW`) | `root` |
| `ids-dashboard` | Flask live dashboard on port 5000 | `adrian` |

### Customise before installing

Edit the unit files in `deploy/` to match your paths:

```bash
# deploy/ids-api.service - adjust WorkingDirectory and venv path
# deploy/clawdbot-agent.service - adjust CLAWDBOT_INTERFACE, paths

# For Telegram credentials, create a .env file:
cat > /home/adrian/fresh_start/.env << 'EOF'
TELEGRAM_BOT_TOKEN=7123456789:AAHxxxxxx
TELEGRAM_CHAT_ID=-100xxxxxxxxxx
EOF
chmod 600 /home/adrian/fresh_start/.env
```

### Management commands

```bash
# View logs
journalctl -u ids-api -f
journalctl -u clawdbot-agent -f
journalctl -u ids-dashboard -f

# Restart individual services
sudo systemctl restart ids-api
sudo systemctl restart clawdbot-agent
sudo systemctl restart ids-dashboard

# Stop everything
sudo systemctl stop clawdbot-agent ids-dashboard ids-api
```

---

## Restarting services

### Restart Ollama

```bash
sudo systemctl restart ollama
# or if running manually:
pkill ollama && ollama serve &
```

### Restart IDS API

```bash
pkill -f "uvicorn app.main:app"
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
```

### Restart OpenClaw gateway

```bash
openclaw gateway restart
# if that fails (systemd user scope issue):
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
openclaw gateway restart
# or kill and restart manually:
pkill -f "openclaw gateway" && openclaw gateway &
```

### Restart everything

```bash
sudo systemctl restart ollama
pkill -f "uvicorn app.main:app"
pkill -f "openclaw gateway"
sleep 2
source venv_h200/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
openclaw gateway &
```

---

## Environment variables reference

### IDS API

| Variable | Default | Description |
|----------|---------|-------------|
| `TON_IOT_PROJECT_ROOT` | auto-detected | Project root directory |
| `TON_IOT_ARTIFACT_DIR` | `$ROOT/artifacts/resnet_transfer_7class` | Model artifacts path |
| `TON_IOT_MODEL_FILENAME` | `resnet_transfer_model_7class.keras` | Keras model file |
| `TON_IOT_PIPELINE_FILENAME` | `preprocessing_pipeline.pkl` | Preprocessing pipeline |
| `TON_IOT_FEATURES_FILENAME` | `final_features.txt` | Feature list |
| `TON_IOT_API_HOST` | `0.0.0.0` | API bind host |
| `TON_IOT_API_PORT` | `8000` | API bind port |

### Triage backend

| Variable | Default | Description |
|----------|---------|-------------|
| `TON_IOT_TRIAGE_BACKEND` | `ollama` | Backend: `ollama`, `gemini`, or `fallback` |
| `TON_IOT_TRIAGE_TIMEOUT_SECONDS` | `30` | LLM request timeout |

### Ollama

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL_TIER1` | `mistral-small:24b` | Fast model (every alert) |
| `OLLAMA_MODEL_TIER2` | `llama3.1:70b-instruct-q8_0` | Escalation model |
| `OLLAMA_ESCALATION_CONFIDENCE` | `0.75` | Escalate below this confidence |

### Gemini (alternative backend)

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | None | Google API key (required if backend=gemini) |
| `TON_IOT_TRIAGE_MODEL` | `gemini-2.0-flash` | Gemini model name |

### Telegram

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Target chat/group ID |

### ClawdBot capture agent

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAWDBOT_INTERFACE` | **(required)** | Network interface to sniff |
| `CLAWDBOT_BPF_FILTER` | `ip` | BPF packet filter |
| `CLAWDBOT_API_URL` | `http://127.0.0.1:8000` | IDS API base URL |
| `CLAWDBOT_API_TIMEOUT` | `30` | API request timeout (s) |
| `CLAWDBOT_HARVEST_INTERVAL` | `10` | Seconds between flow harvests |
| `CLAWDBOT_SEVERITY_THRESHOLD` | `medium` | Min severity for Telegram alerts |
| `CLAWDBOT_LOG_DIR` | `/data/ton-iot-project/fresh_start/logs` | Attack + action event logs |
| `CLAWDBOT_IGNORE_PORTS` | `22,64295,5000,8000` | Management ports ignored between whitelisted peers before IDS analysis |
| `CLAWDBOT_PROTECTED_IPS` | `100.111.77.70` | Comma-separated protected/server IPs used to normalize attacker -> target roles and exclude server IPs from reputation history |
| `CLAWDBOT_PROTECTED_IPS_FILE` | `$ROOT/data/protected_ips.json` | Dashboard-managed protected IP list reloaded by the agent |
| `CLAWDBOT_FIREWALL_QUEUE` | `$ROOT/data/firewall_requests.json` | Dashboard firewall request queue consumed by the root agent |
| `LOG_LEVEL` | `INFO` | Agent log level |

### Dashboard

| Variable | Default | Description |
|----------|---------|-------------|
| `TON_IOT_DASHBOARD_HOST` | `127.0.0.1` | Dashboard bind host |
| `TON_IOT_DASHBOARD_PORT` | `5000` | Dashboard port |
| `TON_IOT_DASHBOARD_API_URL` | `http://127.0.0.1:8000` | IDS API base URL for status/metadata |
| `TON_IOT_DASHBOARD_LOG_DIR` | `$CLAWDBOT_LOG_DIR` or `$ROOT/logs` | ClawdBot event log directory |
| `TON_IOT_DASHBOARD_AUDIT_LOG` | `$ROOT/artifacts/audit/analyze_events.jsonl` | IDS audit JSONL path |
| `TON_IOT_DASHBOARD_THREAT_DB` | `$ROOT/data/threat_cache.db` | SQLite threat-intelligence cache displayed on the IP Intel page |
| `TON_IOT_DASHBOARD_REFRESH_SECONDS` | `5` | Browser polling interval |
| `TON_IOT_DASHBOARD_IGNORE_PORTS` | `22,64295,5000,8000` | Ports hidden from dashboard metrics; set to `none` to disable |
| `TON_IOT_DASHBOARD_PROTECTED_IPS` | `$CLAWDBOT_PROTECTED_IPS` or `100.111.77.70` | Comma-separated protected/server IPs used for Top Originators/Top Targets role normalization and IP Intel filtering |
| `TON_IOT_DASHBOARD_PROTECTED_IPS_FILE` | `$ROOT/data/protected_ips.json` | Editable protected IP list used by the dashboard and agent |
| `TON_IOT_DASHBOARD_FIREWALL_QUEUE` | `$ROOT/data/firewall_requests.json` | Dashboard queue for manual block/unblock requests |

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info + route list |
| `GET` | `/health` | Health check |
| `GET` | `/metadata` | Model metadata (classes, features, dimensions) |
| `POST` | `/predict` | Classify records (model only) |
| `POST` | `/analyze` | Classify + MITRE triage + audit log |
| `GET` | `/docs` | Interactive Swagger UI |

### Example predict request

```json
{
  "records": [
    {
      "duration": 12,
      "src_bytes": 442,
      "dst_bytes": 1290,
      "proto": "tcp"
    }
  ]
}
```

### Example analyze request (with context)

```json
{
  "records": [
    {
      "duration": 12,
      "src_bytes": 442,
      "dst_bytes": 1290,
      "proto": "tcp"
    }
  ],
  "context": {
    "source": "clawdbot",
    "incident_id": "abc-123"
  }
}
```

Required fields per record: `duration` (numeric >= 0), `src_bytes` (numeric >= 0), `dst_bytes` (numeric >= 0), `proto` (string).

---

## Tiered LLM escalation

The triage service uses a two-tier architecture:

| Tier | Model | VRAM | Speed | When |
|------|-------|------|-------|------|
| **Tier-1** | `mistral-small:24b` | ~26 GB | ~100 tok/s | Every non-normal prediction |
| **Tier-2** | `llama3.1:70b-instruct-q8_0` | ~75 GB | ~40 tok/s | Escalation only |

**Escalation triggers** (automatic):
- Classifier confidence < `OLLAMA_ESCALATION_CONFIDENCE` (default 0.75)
- Triage severity is `review` or `critical`

**Fallback chain:**
1. Tier-1 response -> use it (95% of cases)
2. Tier-1 + escalation -> Tier-2 response replaces Tier-1
3. Tier-2 fails -> keep Tier-1 result
4. Tier-1 fails -> heuristic MITRE mapping (no LLM)

---

## Telegram alert format

```
[HIGH] HIGH SEVERITY - ddos_dos

Detection context
  Classifier label: ddos_dos
  Confidence: 0.923
  Model route: live_cic (0.881)
  Confidence note: Model confidence is high.
  Triage source: ollama:mistral-small:24b

Flow
  TCP 192.168.1.105:51544 -> 10.0.0.1:443

MITRE ATT&CK mapping
  Tactics: Impact
  Technique: T1498 - Network Denial of Service (confidence high; flood behavior)

Analyst summary
  DDoS traffic pattern detected against the HTTPS service.

IP reputation
  Badge: Suspicious (3 hit(s))
  Signals: local severity 2; AbuseIPDB 42

Firewall action
  Blocked 192.168.1.105 for 60min

Next actions:
  1. Block source IP at perimeter firewall
  2. Check for amplification reflectors in network
  3. Escalate to Tier-2 if sustained > 5 min
```

### Severity -> alert mapping

| Severity | Alert | Action |
|----------|-------|--------|
| `low` | No alert (logged only) | - |
| `medium` | [WARN] Warning message | SOC review |
| `high` | [HIGH] Alert message | Immediate triage |
| `critical` | [CRITICAL] Alert + @mention on-call | Incident response |
| `review` | [REVIEW] Review message | Manual classification needed |

---

## Running tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_triage.py -v

# Smoke test against a running API
python helper/smoke_test_api.py
python helper/smoke_test_api.py --base-url http://SERVER_IP:8000
```

---

## LLM model recommendation

### Recommended: Mistral Small 3.1 24B

| Aspect | Detail |
|--------|--------|
| **Model** | `mistral-small:24b` (instruction-tuned) |
| **VRAM (Q8)** | ~26 GB |
| **Speed on H200** | ~80-120 tok/s |
| **Context window** | 128K tokens |
| **Why** | Best speed/accuracy balance for structured MITRE labeling |

### Alternatives

| Model | Size | VRAM (Q8) | Tradeoff |
|-------|------|-----------|----------|
| Qwen 2.5 32B | 32B | ~34 GB | Better reasoning, similar speed |
| Llama 3.1 70B | 70B | ~75 GB | More accurate on edge cases, ~2x slower |
| Phi-4 14B | 14B | ~15 GB | Fastest, good enough for simple labeling |

Both Tier-1 (24B) and Tier-2 (70B) fit simultaneously on H200 at ~101 GB combined, with headroom for the IDS model.
