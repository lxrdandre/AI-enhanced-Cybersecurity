"""Generate synthetic instruction-tuning dataset for ClawdBot SOC triage LoRA.

Produces conversation-formatted JSONL (system / user / assistant turns)
that mirrors the exact prompt structure of ``TriageService._build_prompt()``.

Usage::

    python -m lora.build_dataset --out data/triage_train.jsonl --n-per-class 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# -- Constants - must match TriageService exactly -------------------------

SYSTEM_PROMPT = (
    "You are a SOC analyst assistant. Respond ONLY with valid JSON "
    "matching the requested output_schema. No markdown fences, no commentary."
)

OUTPUT_SCHEMA = {
    "label": "string",
    "severity": "low|medium|high|critical|review",
    "mitre_tactics": ["string"],
    "mitre_techniques": [
        {
            "id": "Txxxx",
            "name": "string",
            "confidence": "high|medium|low",
            "reason": "string",
        }
    ],
    "summary": "string",
    "next_actions": ["string"],
    "confidence_note": "string",
}

ALL_CLASSES = ["backdoor", "ddos_dos", "injection", "normal", "password", "scanning", "xss"]
TRIAGE_CLASSES = ["backdoor", "ddos_dos", "injection", "password", "scanning", "xss"]

# -- MITRE tactics per class ----------------------------------------------

TACTICS: dict[str, list[str]] = {
    "backdoor": ["Persistence", "Command and Control"],
    "ddos_dos": ["Impact"],
    "injection": ["Initial Access", "Execution"],
    "password": ["Credential Access"],
    "scanning": ["Reconnaissance", "Discovery"],
    "xss": ["Initial Access", "Execution"],
}

# -- MITRE techniques per class (pool to sample from) --------------------

TECHNIQUES: dict[str, list[dict]] = {
    "backdoor": [
        {"id": "T1071", "name": "Application Layer Protocol",
         "reason": "C2 traffic using standard application protocol on port {dst_port}."},
        {"id": "T1071.001", "name": "Web Protocols",
         "reason": "HTTP/HTTPS channel on port {dst_port} used for command-and-control."},
        {"id": "T1573", "name": "Encrypted Channel",
         "reason": "Encrypted session on port {dst_port} concealing C2 traffic."},
        {"id": "T1105", "name": "Ingress Tool Transfer",
         "reason": "Asymmetric traffic ({dst_bytes}B received) suggests tool delivery."},
        {"id": "T1041", "name": "Exfiltration Over C2 Channel",
         "reason": "Bidirectional session may include exfiltrated data ({src_bytes}B sent)."},
        {"id": "T1095", "name": "Non-Application Layer Protocol",
         "reason": "Non-standard port {dst_port} used for raw TCP tunnel."},
    ],
    "ddos_dos": [
        {"id": "T1498", "name": "Network Denial of Service",
         "reason": "Volumetric flood targeting port {dst_port} with {src_pkts} packets."},
        {"id": "T1498.001", "name": "Direct Network Flood",
         "reason": "Direct SYN/TCP flood saturating target ({src_bytes}B total)."},
        {"id": "T1498.002", "name": "Reflection Amplification",
         "reason": "Response ({dst_bytes}B) exceeds request ({src_bytes}B), ~{amp_factor:.0f}x amplification."},
        {"id": "T1499", "name": "Endpoint Denial of Service",
         "reason": "Attack targets service on port {dst_port} to exhaust resources."},
        {"id": "T1499.001", "name": "OS Exhaustion Flood",
         "reason": "conn_state {conn_state} and high packet rate exhaust OS connection table."},
    ],
    "injection": [
        {"id": "T1190", "name": "Exploit Public-Facing Application",
         "reason": "Oversized request ({src_bytes}B) to port {dst_port} exploiting input validation."},
        {"id": "T1059", "name": "Command and Scripting Interpreter",
         "reason": "Injected payload likely contains shell commands; {duration:.1f}s session suggests execution."},
        {"id": "T1059.004", "name": "Unix Shell",
         "reason": "Command injection targeting Unix shell via web application on port {dst_port}."},
        {"id": "T1505.003", "name": "Web Shell",
         "reason": "POST followed by interactive session may indicate web shell on port {dst_port}."},
    ],
    "password": [
        {"id": "T1110", "name": "Brute Force",
         "reason": "Repeated short sessions to {service} on port {dst_port}."},
        {"id": "T1110.001", "name": "Password Guessing",
         "reason": "Sequential login attempts typical of online password guessing against {service}."},
        {"id": "T1110.003", "name": "Password Spraying",
         "reason": "Authentication attempts with varied credentials against {service} on port {dst_port}."},
        {"id": "T1078", "name": "Valid Accounts",
         "reason": "Successful auth after brute-force may indicate compromised credentials."},
    ],
    "scanning": [
        {"id": "T1595", "name": "Active Scanning",
         "reason": "Systematic probe to port {dst_port} ({conn_state}) with minimal data exchange."},
        {"id": "T1595.001", "name": "Scanning IP Blocks",
         "reason": "Broad network sweep probing multiple hosts systematically."},
        {"id": "T1595.002", "name": "Vulnerability Scanning",
         "reason": "Service interaction on port {dst_port} indicates vulnerability assessment."},
        {"id": "T1046", "name": "Network Service Discovery",
         "reason": "Service enumeration via {proto} probes ({dst_bytes}B banner response)."},
    ],
    "xss": [
        {"id": "T1189", "name": "Drive-by Compromise",
         "reason": "Injected script in response may execute in victim browsers."},
        {"id": "T1059.007", "name": "JavaScript",
         "reason": "XSS payload embeds JavaScript for client-side execution."},
        {"id": "T1185", "name": "Browser Session Hijacking",
         "reason": "XSS payload may exfiltrate session cookies for account takeover."},
    ],
}

# -- Flow scenario templates ---------------------------------------------
# Each scenario defines parameter ranges for realistic flow generation.
# Tuples are (min, max) ranges; lists are choices.

SCENARIOS: dict[str, list[dict]] = {
    "backdoor": [
        {"dst_ports": [443, 8443], "proto": "tcp", "service": "ssl",
         "conn_state": "S1", "duration": (60, 600),
         "src_bytes": (500, 5000), "dst_bytes": (200, 3000),
         "src_pkts": (10, 100), "dst_pkts": (5, 50),
         "summary": "Persistent encrypted connection on port {dst_port} lasting {duration:.0f}s. "
                    "Bidirectional traffic ({src_bytes}B\u2191/{dst_bytes}B\u2193) consistent with C2 beaconing over HTTPS."},
        {"dst_ports": [4444, 1337, 9999, 5555], "proto": "tcp", "service": "-",
         "conn_state": "S1", "duration": (30, 300),
         "src_bytes": (100, 2000), "dst_bytes": (1000, 10000),
         "src_pkts": (5, 50), "dst_pkts": (10, 100),
         "summary": "Long-lived TCP session on non-standard port {dst_port} ({duration:.0f}s). "
                    "Asymmetric traffic: {dst_bytes}B received vs {src_bytes}B sent indicates interactive reverse shell."},
        {"dst_ports": [80, 8080], "proto": "tcp", "service": "http",
         "conn_state": "SF", "duration": (5, 120),
         "src_bytes": (200, 3000), "dst_bytes": (500, 8000),
         "src_pkts": (3, 30), "dst_pkts": (3, 30), "http_method": "POST",
         "summary": "HTTP session to port {dst_port} with structured request-response pattern. "
                    "Response payload ({dst_bytes}B) disproportionate to request, consistent with C2 command delivery."},
    ],
    "ddos_dos": [
        {"dst_ports": [80, 443, 22, 53], "proto": "tcp",
         "service_choices": ["http", "ssl", "ssh", "dns"],
         "conn_state": "S0", "duration": (0, 2),
         "src_bytes": (40, 120), "dst_bytes": (0, 0),
         "src_pkts": (1, 5), "dst_pkts": (0, 0),
         "summary": "SYN flood probe to port {dst_port}: {src_pkts} unanswered SYN packets "
                    "(conn_state S0). Volumetric denial-of-service indicator."},
        {"dst_ports": [80, 443, 8080], "proto": "tcp",
         "service_choices": ["http", "ssl", "http"],
         "conn_state": "SF", "duration": (0.5, 10),
         "src_bytes": (5000, 50000), "dst_bytes": (1000, 20000),
         "src_pkts": (50, 500), "dst_pkts": (30, 300), "http_method": "GET",
         "summary": "HTTP flood against port {dst_port}: {src_pkts} requests ({src_bytes}B) in "
                    "{duration:.1f}s. Elevated rate risks server resource exhaustion."},
        {"dst_ports": [53, 123, 161], "proto": "udp",
         "service_choices": ["dns", "-", "-"],
         "conn_state": "OTH", "duration": (0.1, 5),
         "src_bytes": (50, 500), "dst_bytes": (2000, 50000),
         "src_pkts": (1, 10), "dst_pkts": (5, 100),
         "summary": "UDP amplification via port {dst_port}: {src_bytes}B request \u2192 {dst_bytes}B response "
                    "(~{amp_factor:.0f}x amplification). Reflection/amplification attack."},
    ],
    "injection": [
        {"dst_ports": [80, 443, 8080], "proto": "tcp", "service": "http",
         "conn_state": "SF", "duration": (0.5, 10),
         "src_bytes": (500, 5000), "dst_bytes": (1000, 50000),
         "src_pkts": (3, 20), "dst_pkts": (5, 30), "http_method": "POST",
         "summary": "SQL injection via HTTP POST to port {dst_port}. Abnormally large "
                    "request ({src_bytes}B) with oversized response ({dst_bytes}B) suggests data extraction."},
        {"dst_ports": [80, 443, 8080, 8443], "proto": "tcp", "service": "http",
         "conn_state": "SF", "duration": (1, 30),
         "src_bytes": (200, 3000), "dst_bytes": (500, 20000),
         "src_pkts": (2, 15), "dst_pkts": (3, 20), "http_method": "POST",
         "summary": "Command injection via HTTP POST to port {dst_port}. Extended session "
                    "({duration:.1f}s) and response size ({dst_bytes}B) indicate server-side execution."},
        {"dst_ports": [3306, 5432, 1433], "proto": "tcp",
         "service_choices": ["mysql", "postgres", "-"],
         "conn_state": "SF", "duration": (0.5, 20),
         "src_bytes": (200, 5000), "dst_bytes": (1000, 100000),
         "src_pkts": (5, 30), "dst_pkts": (10, 50),
         "summary": "Direct database injection on port {dst_port}: oversized query ({src_bytes}B) "
                    "with bulk response ({dst_bytes}B) indicating unauthorized data retrieval."},
    ],
    "password": [
        {"dst_ports": [22], "proto": "tcp", "service": "ssh",
         "conn_state_choices": ["SF", "REJ", "RSTR"], "duration": (0.1, 5),
         "src_bytes": (100, 500), "dst_bytes": (100, 500),
         "src_pkts": (3, 15), "dst_pkts": (3, 15),
         "summary": "SSH brute-force attempt on port {dst_port} (conn_state {conn_state}). "
                    "Short session ({duration:.1f}s) typical of automated credential guessing."},
        {"dst_ports": [21], "proto": "tcp", "service": "ftp",
         "conn_state_choices": ["SF", "REJ"], "duration": (0.1, 3),
         "src_bytes": (50, 300), "dst_bytes": (50, 400),
         "src_pkts": (2, 10), "dst_pkts": (2, 10),
         "summary": "FTP brute-force on port {dst_port}. Rapid connect/disconnect ({duration:.1f}s) "
                    "indicates credential stuffing against FTP service."},
        {"dst_ports": [80, 443, 8080], "proto": "tcp", "service": "http",
         "conn_state": "SF", "duration": (0.2, 5),
         "src_bytes": (200, 1000), "dst_bytes": (500, 3000),
         "src_pkts": (2, 10), "dst_pkts": (2, 10), "http_method": "POST",
         "summary": "HTTP login brute-force via POST to port {dst_port}. Request pattern "
                    "({src_bytes}B) consistent with automated web credential testing."},
    ],
    "scanning": [
        {"dst_ports": list(range(20, 1025)), "proto": "tcp", "service": "-",
         "conn_state_choices": ["S0", "REJ", "RSTR"], "duration": (0, 0.5),
         "src_bytes": (40, 80), "dst_bytes": (0, 60),
         "src_pkts": (1, 3), "dst_pkts": (0, 2),
         "summary": "TCP SYN scan to port {dst_port} (conn_state {conn_state}). "
                    "Single probe with {duration:.3f}s timeout indicates port enumeration."},
        {"dst_ports": [22, 80, 443, 21, 25, 3306, 5432, 8080], "proto": "tcp",
         "service_choices": ["ssh", "http", "ssl", "ftp", "smtp", "mysql", "postgres", "http"],
         "conn_state": "SF", "duration": (0.5, 5),
         "src_bytes": (100, 500), "dst_bytes": (200, 2000),
         "src_pkts": (2, 10), "dst_pkts": (2, 10),
         "summary": "Service version probe on port {dst_port} ({service}). Established connection "
                    "with banner grab ({dst_bytes}B) indicates service fingerprinting."},
        {"dst_ports": [53, 123, 161, 500, 1900], "proto": "udp",
         "service_choices": ["dns", "-", "-", "-", "-"],
         "conn_state": "OTH", "duration": (0, 2),
         "src_bytes": (30, 100), "dst_bytes": (0, 200),
         "src_pkts": (1, 3), "dst_pkts": (0, 2),
         "summary": "UDP probe to port {dst_port}: minimal payload ({src_bytes}B). "
                    "Reconnaissance mapping open UDP services."},
    ],
    "xss": [
        {"dst_ports": [80, 443, 8080], "proto": "tcp", "service": "http",
         "conn_state": "SF", "duration": (0.2, 5),
         "src_bytes": (300, 2000), "dst_bytes": (1000, 10000),
         "src_pkts": (2, 10), "dst_pkts": (3, 15), "http_method": "GET",
         "summary": "Reflected XSS via HTTP GET to port {dst_port}. Oversized URL ({src_bytes}B) "
                    "contains injected script. Response ({dst_bytes}B) may reflect unsanitized input."},
        {"dst_ports": [80, 443, 8080], "proto": "tcp", "service": "http",
         "conn_state": "SF", "duration": (0.5, 10),
         "src_bytes": (500, 5000), "dst_bytes": (500, 5000),
         "src_pkts": (2, 10), "dst_pkts": (3, 15), "http_method": "POST",
         "summary": "Stored XSS via HTTP POST to port {dst_port}. Form submission ({src_bytes}B) "
                    "embeds JavaScript payload for persistent storage and later execution."},
        {"dst_ports": [80, 443], "proto": "tcp", "service": "http",
         "conn_state": "SF", "duration": (0.1, 3),
         "src_bytes": (200, 1500), "dst_bytes": (2000, 15000),
         "src_pkts": (1, 5), "dst_pkts": (3, 20), "http_method": "GET",
         "summary": "DOM-based XSS probe to port {dst_port}. GET request with crafted parameters "
                    "({src_bytes}B) targeting client-side rendering. Large response ({dst_bytes}B)."},
    ],
}

# -- Next-action pools per class ------------------------------------------

NEXT_ACTIONS: dict[str, list[str]] = {
    "backdoor": [
        "Inspect endpoint for persistence mechanisms (scheduled tasks, startup items).",
        "Capture full PCAP of C2 session for payload analysis and IOC extraction.",
        "Isolate affected host from network and initiate forensic investigation.",
        "Check DNS logs for DGA-like domain patterns from this source.",
        "Review outbound connections for signs of data exfiltration.",
        "Cross-reference source IP with threat intelligence feeds.",
        "Scan endpoint for known RAT/backdoor signatures.",
    ],
    "ddos_dos": [
        "Enable rate limiting or traffic shaping on affected service.",
        "Contact upstream ISP if volumetric attack exceeds local mitigation capacity.",
        "Activate DDoS mitigation (null-route or scrubbing center).",
        "Monitor service availability and response times for degradation.",
        "Check for reflector/amplifier systems within own network.",
        "Coordinate with NOC for traffic engineering response.",
    ],
    "injection": [
        "Review web application logs for injected payloads and attack patterns.",
        "Audit database query logs for unauthorized data access.",
        "Inspect application input validation and parameterized query usage.",
        "Check WAF rules triggered and update signatures if needed.",
        "Assess scope of potential data breach from successful injection.",
        "Deploy application-layer monitoring on affected endpoint.",
    ],
    "password": [
        "Review authentication logs for failed/successful logins from this source.",
        "Verify account lockout policies are enforced and triggered.",
        "Check if any credentials were successfully compromised.",
        "Implement or verify MFA on targeted accounts.",
        "Consider IP-based rate limiting on authentication endpoints.",
        "Rotate credentials for accounts with recent successful logins.",
    ],
    "scanning": [
        "Document scan scope (ports, protocols, timing) for threat assessment.",
        "Verify if source is authorized pentest or expected scan activity.",
        "Monitor for exploitation attempts following reconnaissance.",
        "Restrict unnecessary open ports identified during scan response.",
        "Update IDS signatures for scan patterns from this source.",
        "Consider blocking source IP if scanning persists.",
    ],
    "xss": [
        "Review web application for unescaped user input in HTML output.",
        "Audit Content-Security-Policy headers on affected pages.",
        "Check web proxy logs for evidence of payload execution.",
        "Implement output encoding and input sanitization on affected endpoints.",
        "Scan for other injection points in the same application.",
        "Review access logs for evidence of session cookie theft.",
    ],
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
    "python-requests/2.31.0",
    "curl/8.4.0",
    "Go-http-client/2.0",
    "-",
]

DNS_DOMAINS = [
    "update-check.example.com", "cdn-assets.example.net", "api.malware-c2.xyz",
    "ns1.evil-dns.com", "tracker.botnet.ru", "data.exfil-server.io",
]

# -- Generation helpers ---------------------------------------------------


def _ri(rng: random.Random, r: tuple) -> int:
    """Random int in [r[0], r[1]]."""
    return rng.randint(int(r[0]), int(r[1]))


def _rf(rng: random.Random, r: tuple) -> float:
    """Random float in [r[0], r[1]]."""
    return round(rng.uniform(float(r[0]), float(r[1])), 6)


def _generate_flow(scenario: dict, rng: random.Random) -> dict:
    """Generate a single 25-feature flow record from a scenario template."""
    # Pick port and service together
    if "service_choices" in scenario:
        idx = rng.randrange(len(scenario["service_choices"]))
        dst_port = scenario["dst_ports"][idx % len(scenario["dst_ports"])]
        service = scenario["service_choices"][idx]
    else:
        dst_port = rng.choice(scenario["dst_ports"])
        service = scenario.get("service", "-")

    src_port = rng.randint(32768, 65535)
    proto = scenario["proto"]

    if "conn_state_choices" in scenario:
        conn_state = rng.choice(scenario["conn_state_choices"])
    else:
        conn_state = scenario.get("conn_state", "OTH")

    duration = _rf(rng, scenario["duration"])
    src_bytes = _ri(rng, scenario["src_bytes"])
    dst_bytes = _ri(rng, scenario["dst_bytes"])
    src_pkts = _ri(rng, scenario["src_pkts"])
    dst_pkts = _ri(rng, scenario["dst_pkts"])

    # HTTP fields - populate when relevant
    is_http = service in ("http",) or "http_method" in scenario
    http_method = scenario.get("http_method", rng.choice(["GET", "POST"])) if is_http else "-"
    http_ua = rng.choice(USER_AGENTS) if is_http else "-"
    http_version = rng.choice(["1.1", "2.0"]) if is_http else "-"
    http_resp_len = max(0, dst_bytes - rng.randint(50, 200)) if is_http else 0
    http_status = rng.choice([200, 200, 200, 301, 403, 500]) if is_http else 0
    http_depth = rng.randint(1, 5) if is_http else 0

    # DNS fields - populate for port 53
    dns_query = "-"
    dns_qclass = "-"
    dns_qtype = "-"
    dns_rcode = "-"
    dns_aa = 0
    if dst_port == 53:
        dns_query = rng.choice(DNS_DOMAINS)
        dns_qclass = "1"
        dns_qtype = rng.choice(["1", "28", "255"])
        dns_rcode = "0"
        dns_aa = rng.randint(0, 1)

    return {
        "src_port": src_port,
        "dst_port": dst_port,
        "proto": proto,
        "service": service,
        "duration": duration,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "conn_state": conn_state,
        "missed_bytes": 0,
        "src_pkts": src_pkts,
        "src_ip_bytes": src_bytes + src_pkts * 20,
        "dst_pkts": dst_pkts,
        "dst_ip_bytes": dst_bytes + dst_pkts * 20,
        "dns_query": dns_query,
        "dns_qclass": dns_qclass,
        "dns_qtype": dns_qtype,
        "dns_rcode": dns_rcode,
        "dns_AA": dns_aa,
        "http_trans_depth": http_depth,
        "http_method": http_method,
        "http_referrer": "-",
        "http_version": http_version,
        "http_response_body_len": http_resp_len,
        "http_status_code": http_status,
        "http_user_agent": http_ua,
    }


def _generate_prediction(cls: str, rng: random.Random) -> dict:
    """Generate classifier prediction with realistic probability distribution."""
    # Diverse confidence range: 25% each bucket
    confidence = rng.choice([
        rng.uniform(0.50, 0.65),
        rng.uniform(0.65, 0.80),
        rng.uniform(0.80, 0.92),
        rng.uniform(0.92, 0.99),
    ])
    confidence = round(confidence, 4)

    # Distribute remainder among other classes
    remainder = 1.0 - confidence
    others = [c for c in ALL_CLASSES if c != cls]
    raw = [rng.random() for _ in others]
    total = sum(raw)
    probs = {c: round(r / total * remainder, 6) for c, r in zip(others, raw)}
    probs[cls] = confidence

    # Fix rounding drift
    diff = round(1.0 - sum(probs.values()), 6)
    probs[cls] = round(probs[cls] + diff, 6)

    return {
        "predicted_index": ALL_CLASSES.index(cls),
        "predicted_label": cls,
        "confidence": confidence,
        "probabilities": probs,
    }


def _severity_for(cls: str, confidence: float, rng: random.Random) -> str:
    """Determine severity from class and confidence with some variance."""
    if confidence >= 0.90:
        if cls in ("backdoor", "injection"):
            return rng.choices(["critical", "high"], weights=[0.3, 0.7])[0]
        if cls == "ddos_dos":
            return rng.choices(["high", "critical"], weights=[0.6, 0.4])[0]
        return rng.choices(["high", "medium"], weights=[0.7, 0.3])[0]
    if confidence >= 0.75:
        return rng.choices(["medium", "high"], weights=[0.6, 0.4])[0]
    if confidence >= 0.60:
        return rng.choices(["medium", "review"], weights=[0.5, 0.5])[0]
    return "review"


def _confidence_note(confidence: float) -> str:
    """Return a short confidence-quality explanation."""
    if confidence >= 0.95:
        return f"Very high model confidence ({confidence:.1%}). Detection is highly reliable."
    if confidence >= 0.85:
        return f"High model confidence ({confidence:.1%}). Detection is reliable."
    if confidence >= 0.75:
        return (
            f"Moderate model confidence ({confidence:.1%}). "
            "Corroborating evidence recommended before escalation."
        )
    if confidence >= 0.60:
        return (
            f"Low-moderate confidence ({confidence:.1%}). "
            "Manual review strongly recommended before action."
        )
    return (
        f"Low model confidence ({confidence:.1%}). "
        "Detection requires human validation; may be false positive."
    )


def _generate_triage(
    cls: str, scenario: dict, flow: dict, prediction: dict, rng: random.Random,
) -> dict:
    """Generate gold-standard triage response."""
    confidence = prediction["confidence"]
    severity = _severity_for(cls, confidence, rng)

    # Pick 1-3 techniques from class pool
    pool = TECHNIQUES[cls]
    n = min(rng.randint(1, 3), len(pool))
    selected = rng.sample(pool, n)

    amp_factor = flow["dst_bytes"] / max(flow["src_bytes"], 1)
    fmt = {**flow, "amp_factor": amp_factor}

    techniques = []
    for t in selected:
        tc = "high" if confidence >= 0.85 else "medium" if confidence >= 0.70 else "low"
        techniques.append({
            "id": t["id"],
            "name": t["name"],
            "confidence": tc,
            "reason": t["reason"].format(**fmt),
        })

    summary = scenario["summary"].format(**fmt)
    actions = rng.sample(NEXT_ACTIONS[cls], min(rng.randint(2, 4), len(NEXT_ACTIONS[cls])))

    return {
        "label": cls,
        "severity": severity,
        "mitre_tactics": TACTICS[cls],
        "mitre_techniques": techniques,
        "summary": summary,
        "next_actions": actions,
        "confidence_note": _confidence_note(confidence),
    }


def _build_prompt(prediction: dict, record: dict) -> str:
    """Build user prompt - mirrors ``TriageService._build_prompt()`` exactly."""
    prompt = {
        "task": "SOC triage and MITRE enrichment",
        "constraints": [
            "Do not relabel classifier output.",
            "Return JSON only.",
            "If uncertain, set severity to review.",
            "Include MITRE tactics, techniques, confidence, and reason fields that explain the mapping.",
            "Make next_actions concrete enough for a first responder to execute.",
        ],
        "output_schema": OUTPUT_SCHEMA,
        "classifier_prediction": prediction,
        "record": record,
        "context": {},
    }
    return json.dumps(prompt, ensure_ascii=True)


# -- Dataset assembly -----------------------------------------------------


def generate_dataset(
    n_per_class: int, seed: int, test_ratio: float = 0.15,
) -> tuple[list[dict], list[dict]]:
    """Generate train/test split of conversation-formatted examples."""
    rng = random.Random(seed)
    examples: list[dict] = []

    for cls in TRIAGE_CLASSES:
        scenarios = SCENARIOS[cls]
        for _ in range(n_per_class):
            scenario = rng.choice(scenarios)
            flow = _generate_flow(scenario, rng)
            prediction = _generate_prediction(cls, rng)
            triage = _generate_triage(cls, scenario, flow, prediction, rng)

            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_prompt(prediction, flow)},
                    {"role": "assistant", "content": json.dumps(triage, ensure_ascii=False)},
                ],
            })

    rng.shuffle(examples)
    split = max(1, int(len(examples) * (1 - test_ratio)))
    return examples[:split], examples[split:]


def _write_jsonl(path: str, data: list[dict]) -> None:
    """Write jsonl."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic triage training data")
    parser.add_argument("--out", default="data/triage_train.jsonl", help="Training JSONL path")
    parser.add_argument("--test-out", default="data/triage_test.jsonl", help="Test JSONL path")
    parser.add_argument("--n-per-class", type=int, default=50, help="Examples per attack class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_data, test_data = generate_dataset(args.n_per_class, args.seed)

    _write_jsonl(args.out, train_data)
    _write_jsonl(args.test_out, test_data)

    print(f"Train: {len(train_data)} examples  \u2192 {args.out}")
    print(f"Test:  {len(test_data)} examples  \u2192 {args.test_out}")

    counts = Counter()
    for ex in train_data:
        user_msg = json.loads(ex["messages"][1]["content"])
        counts[user_msg["classifier_prediction"]["predicted_label"]] += 1
    print(f"\nTraining distribution: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
