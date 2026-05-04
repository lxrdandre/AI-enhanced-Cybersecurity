const refreshSeconds = Number(document.body.dataset.refresh || 5);
const colors = ["#45f0b1", "#78a8ff", "#ffbd5a", "#ff5f6d", "#b38cff", "#5ee1ff", "#d9f99d", "#f8a5c2"];
const INCIDENT_PAGE_SIZE = 20;
const TELEGRAM_SEEN_STORAGE_KEY = "toniot_seen_telegram_alerts_v1";
const INITIAL_ALERT_WINDOW_SECONDS = Math.max(30, refreshSeconds * 3);
const MAX_VISIBLE_TELEGRAM_TOASTS = 5;
let refreshInFlight = false;
let visibleIncidentCount = INCIDENT_PAGE_SIZE;
let latestIncidentRows = [];
let incidentFilters = { search: "", severity: "", label: "" };
let telegramAlertsInitialized = false;
let seenTelegramAlerts = loadSeenTelegramAlerts();
const chartInstances = {};

/**
 * Format numeric values for compact dashboard display.
 */
function fmt(value) {
  return new Intl.NumberFormat().format(Number(value || 0));
}

/**
 * Format a count as the dashboard possible-threat phrase.
 */
function possiblePhrase(value) {
  const count = Number(value || 0);
  return `${fmt(count)} possible`;
}

/**
 * Return a compact relative age string.
 */
function relativeAge(epoch) {
  const value = Number(epoch || 0);
  if (!value) return "never";
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - value));
  if (seconds < 5) return "now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

/**
 * Set text content for an element when it exists.
 */
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

/**
 * Convert a severity value into a safe CSS class name.
 */
function severityClass(severity) {
  return String(severity || "unknown").toLowerCase().replace(/[^a-z0-9_-]/g, "");
}

/**
 * Escape text before inserting it into HTML.
 */
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/**
 * Load seen telegram alerts from browser or API state.
 */
function loadSeenTelegramAlerts() {
  try {
    const values = JSON.parse(sessionStorage.getItem(TELEGRAM_SEEN_STORAGE_KEY) || "[]");
    return new Set(Array.isArray(values) ? values : []);
  } catch {
    return new Set();
  }
}

/**
 * Save seen telegram alerts to browser state.
 */
function saveSeenTelegramAlerts() {
  try {
    const values = Array.from(seenTelegramAlerts).slice(-300);
    seenTelegramAlerts = new Set(values);
    sessionStorage.setItem(TELEGRAM_SEEN_STORAGE_KEY, JSON.stringify(values));
  } catch {
    // Session storage can be disabled; in-memory tracking is enough for this tab.
  }
}

/**
 * Build a stable key for incident rows and Telegram popups.
 */
function incidentKey(row) {
  return String(row?.event_id || [
    row?.audit_id,
    row?.time,
    row?.label,
    row?.src,
    row?.dst,
    row?.port,
    row?.confidence,
  ].join("|"));
}

/**
 * Return chart colors with an optional alpha suffix.
 */
function chartColors(opacity = 1) {
  return colors.map(color => opacity === 1 ? color : `${color}${Math.round(opacity * 255).toString(16).padStart(2, "0")}`);
}

/**
 * Return shared Chart.js options for dashboard charts.
 */
function baseChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    resizeDelay: 120,
    devicePixelRatio: Math.min(window.devicePixelRatio || 1, 2),
    animation: { duration: 280 },
    layout: { padding: 4 },
    plugins: {
      legend: {
        labels: {
          color: "#b6c9c2",
          boxWidth: 10,
          boxHeight: 10,
          usePointStyle: true,
          font: { family: "IBM Plex Sans", size: 12 },
        },
      },
      tooltip: {
        backgroundColor: "rgba(7, 16, 15, 0.96)",
        borderColor: "rgba(148, 180, 169, 0.25)",
        borderWidth: 1,
        titleColor: "#e8f3ef",
        bodyColor: "#cfe0da",
        padding: 10,
      },
    },
  };
}

/**
 * Create or update a Chart.js instance for a canvas.
 */
function replaceChart(canvasId, config) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return null;
  const existing = chartInstances[canvasId];
  if (existing && existing.config.type === config.type) {
    existing.data = config.data;
    existing.options = config.options;
    existing.update("none");
    return existing;
  }
  chartInstances[canvasId]?.destroy();
  chartInstances[canvasId] = new Chart(canvas, config);
  return chartInstances[canvasId];
}

/**
 * Draw timeline chart data.
 */
function drawTimeline(canvasId, points) {
  const safePoints = Array.isArray(points) ? points : [];
  if (!window.Chart) return;

  replaceChart(canvasId, {
    type: "line",
    data: {
      labels: safePoints.map(point => point.label || ""),
      datasets: [
        {
          label: "All detections",
          data: safePoints.map(point => Number(point.total || 0)),
          borderColor: "#45f0b1",
          backgroundColor: "rgba(69, 240, 177, 0.12)",
          fill: true,
          tension: 0,
          clip: 0,
          pointRadius: 0,
          pointHoverRadius: 4,
          borderWidth: 2.4,
        },
        {
          label: "High + critical",
          data: safePoints.map(point => Number(point.high || 0)),
          borderColor: "#ff5f6d",
          backgroundColor: "rgba(255, 95, 109, 0.10)",
          fill: true,
          tension: 0,
          clip: 0,
          pointRadius: 0,
          pointHoverRadius: 4,
          borderWidth: 2.2,
        },
      ],
    },
    options: {
      ...baseChartOptions(),
      interaction: { intersect: false, mode: "index" },
      scales: {
        x: {
          ticks: { color: "#8fa59d", maxTicksLimit: 8, font: { family: "JetBrains Mono", size: 11 } },
          grid: { color: "rgba(148, 180, 169, 0.08)" },
        },
        y: {
          min: 0,
          beginAtZero: true,
          grace: "8%",
          ticks: { color: "#8fa59d", precision: 0, font: { family: "JetBrains Mono", size: 11 } },
          grid: { color: "rgba(148, 180, 169, 0.14)" },
        },
      },
    },
  });
}

/**
 * Draw donut chart data.
 */
function drawDonut(canvasId, items, emptyLabel = "No data") {
  const safeItems = Array.isArray(items) ? items.filter(item => Number(item.value || 0) > 0) : [];
  if (!window.Chart) return;

  const hasData = safeItems.length > 0;
  replaceChart(canvasId, {
    type: "doughnut",
    data: {
      labels: hasData ? safeItems.map(item => item.name) : [emptyLabel],
      datasets: [{
        data: hasData ? safeItems.map(item => Number(item.value || 0)) : [1],
        backgroundColor: hasData ? chartColors(0.92) : ["rgba(148, 180, 169, 0.18)"],
        borderColor: "rgba(7, 16, 15, 0.85)",
        borderWidth: 2,
        hoverOffset: 8,
      }],
    },
    options: {
      ...baseChartOptions(),
      cutout: "64%",
      plugins: {
        ...baseChartOptions().plugins,
        legend: {
          position: "bottom",
          labels: {
            color: "#b6c9c2",
            boxWidth: 10,
            boxHeight: 10,
            usePointStyle: true,
            padding: 14,
            font: { family: "IBM Plex Sans", size: 12 },
          },
        },
        tooltip: {
          ...baseChartOptions().plugins.tooltip,
          enabled: hasData,
        },
      },
    },
  });
}

/**
 * Render bars into the dashboard DOM.
 */
function renderBars(id, items) {
  const el = document.getElementById(id);
  if (!el) return;
  const max = Math.max(1, ...items.map(item => item.value));
  el.innerHTML = items.length ? items.map((item, index) => `
    <div class="bar-row">
      <span>${escapeHtml(item.name)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(4, item.value / max * 100)}%; background:${colors[index % colors.length]}"></div></div>
      <strong>${escapeHtml(item.value)}</strong>
    </div>
  `).join("") : `<p class="muted">No data yet.</p>`;
}

/**
 * Render rank list into the dashboard DOM.
 */
function renderRankList(id, items) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = items.length ? items.map(item => `
    <div class="rank-row">
      <span class="rank-name">${escapeHtml(item.name)}</span>
      <strong>${escapeHtml(item.value)}</strong>
    </div>
  `).join("") : `<p class="muted">No data yet.</p>`;
}

/**
 * Render system pulse into the dashboard DOM.
 */
function renderSystemPulse(activity = {}, api = {}) {
  const validStates = new Set(["breathing", "thinking", "dead"]);
  const state = validStates.has(activity.state) ? activity.state : (api.online ? "breathing" : "dead");
  const mode = {
    breathing: "STANDBY ONLINE",
    thinking: "ACTIVE INFERENCE",
    dead: "NO SIGNAL",
  }[state];

  const orb = document.getElementById("pulseOrb");
  const pill = document.getElementById("pulseState");
  const signals = document.getElementById("pulseSignals");

  if (orb) {
    orb.classList.remove("breathing", "thinking", "dead");
    orb.classList.add(state);
  }
  if (pill) {
    pill.className = `pill pulse-status ${state}`;
    pill.textContent = String(activity.label || state).toUpperCase();
  }

  setText("pulseHeadline", activity.headline || (api.online ? "Neural core online" : "No signal"));
  setText("pulseMode", mode);
  setText("pulseDetail", activity.detail || "Waiting for dashboard telemetry.");
  setText("pulseAnalysis", relativeAge(activity.last_analysis_epoch));
  setText("pulseAttack", relativeAge(activity.last_attack_epoch));
  setText("pulseAction", relativeAge(activity.last_action_epoch));

  if (signals) {
    const rows = Array.isArray(activity.signals) ? activity.signals : [];
    signals.innerHTML = rows.length ? rows.map(item => `
      <span class="signal-chip">
        <small>${escapeHtml(item.name)}</small>
        <strong>${escapeHtml(item.value)}</strong>
      </span>
    `).join("") : `<span class="signal-chip"><small>Status</small><strong>${escapeHtml(api.status || "unknown")}</strong></span>`;
  }
}

/**
 * Render incidents into the dashboard DOM.
 */
function renderIncidents(rows) {
  const el = document.getElementById("incidentRows");
  if (!el) return;
  latestIncidentRows = Array.isArray(rows) ? rows : [];
  populateIncidentLabelFilter(latestIncidentRows);
  const filteredRows = filterIncidentRows(latestIncidentRows);
  const countEl = document.getElementById("incidentCount");
  const button = document.getElementById("showMoreIncidents");
  const visibleRows = filteredRows.slice(0, visibleIncidentCount);

  if (!filteredRows.length) {
    el.innerHTML = `<tr><td colspan="12">No threat events recorded yet.</td></tr>`;
    if (countEl) countEl.textContent = `Showing 0 of ${latestIncidentRows.length} detections`;
    if (button) button.hidden = true;
    return;
  }

  el.innerHTML = visibleRows.map(row => {
    const index = latestIncidentRows.indexOf(row);
    return `
    <tr class="${row.telegram_sent ? "telegram-row" : ""}">
      <td>${escapeHtml(row.time)}</td>
      <td>${escapeHtml(row.label)}</td>
      <td><span class="severity ${severityClass(row.severity)}">${escapeHtml(row.severity)}</span></td>
      <td><span class="alert-badge ${row.telegram_sent ? "sent" : "logged"}">${row.telegram_sent ? "Telegram" : "Logged"}</span></td>
      <td>${escapeHtml(row.role || "-")}</td>
      <td>${escapeHtml(row.route || "-")}</td>
      <td>${Number(row.confidence || 0).toFixed(3)}</td>
      <td>${escapeHtml(row.src)}</td>
      <td>${escapeHtml(row.dst)}</td>
      <td>${escapeHtml(row.port)}</td>
      <td>${escapeHtml(row.summary || row.source || "-")}</td>
      <td><button class="detail-button" type="button" data-incident-index="${index}">Open</button></td>
    </tr>
  `}).join("");

  const shown = Math.min(visibleRows.length, filteredRows.length);
  if (countEl) countEl.textContent = `Showing ${shown} of ${filteredRows.length} matching detections (${latestIncidentRows.length} total)`;
  if (button) {
    button.hidden = shown >= filteredRows.length;
    button.textContent = `Show ${Math.min(INCIDENT_PAGE_SIZE, filteredRows.length - shown)} more`;
  }
}

/**
 * Filter incident rows using active controls.
 */
function filterIncidentRows(rows) {
  const query = incidentFilters.search.toLowerCase();
  return rows.filter(row => {
    if (incidentFilters.severity && String(row.severity || "").toLowerCase() !== incidentFilters.severity) return false;
    if (incidentFilters.label && String(row.label || "").toLowerCase() !== incidentFilters.label) return false;
    if (!query) return true;
    return [row.label, row.severity, row.src, row.dst, row.port, row.summary, row.route]
      .some(value => String(value || "").toLowerCase().includes(query));
  });
}

/**
 * Populate incident label filter controls.
 */
function populateIncidentLabelFilter(rows) {
  const select = document.getElementById("incidentLabel");
  if (!select) return;
  const current = select.value;
  const labels = Array.from(new Set(rows.map(row => String(row.label || "")).filter(Boolean))).sort();
  select.innerHTML = `<option value="">All labels</option>` + labels.map(label =>
    `<option value="${escapeHtml(label.toLowerCase())}">${escapeHtml(label)}</option>`
  ).join("");
  select.value = labels.map(label => label.toLowerCase()).includes(current) ? current : "";
}

/**
 * Initialize incident filters behavior.
 */
function initIncidentFilters() {
  const search = document.getElementById("incidentSearch");
  const severity = document.getElementById("incidentSeverity");
  const label = document.getElementById("incidentLabel");
  search?.addEventListener("input", () => {
    incidentFilters.search = search.value.trim();
    visibleIncidentCount = INCIDENT_PAGE_SIZE;
    renderIncidents(latestIncidentRows);
  });
  severity?.addEventListener("change", () => {
    incidentFilters.severity = severity.value;
    visibleIncidentCount = INCIDENT_PAGE_SIZE;
    renderIncidents(latestIncidentRows);
  });
  label?.addEventListener("change", () => {
    incidentFilters.label = label.value;
    visibleIncidentCount = INCIDENT_PAGE_SIZE;
    renderIncidents(latestIncidentRows);
  });
}

/**
 * Render telegram toast into the dashboard DOM.
 */
function renderTelegramToast(row) {
  const stack = document.getElementById("telegramToastStack");
  if (!stack) return;
  const key = incidentKey(row);
  const firstTechnique = Array.isArray(row.mitre_techniques) && row.mitre_techniques.length
    ? row.mitre_techniques[0]
    : null;
  const mitreText = firstTechnique
    ? `MITRE ${firstTechnique.id || "unknown"} ${firstTechnique.name || ""}`.trim()
    : "";
  const toast = document.createElement("article");
  toast.className = `telegram-toast ${severityClass(row.severity)}`;
  toast.dataset.alertKey = key;
  toast.innerHTML = `
    <div class="toast-head">
      <span>Telegram Alert</span>
      <button type="button" aria-label="Dismiss alert" data-dismiss-alert>x</button>
    </div>
    <strong>${escapeHtml(row.label)} detected</strong>
    <p>${escapeHtml(row.src || "-")} -> ${escapeHtml(row.dst || "-")}:${escapeHtml(row.port || "-")}</p>
    <small>${escapeHtml(row.severity || "unknown")} severity - confidence ${Number(row.confidence || 0).toFixed(3)} - ${escapeHtml(row.route || "-")}</small>
    ${mitreText ? `<small>${escapeHtml(mitreText)}</small>` : ""}
    <div class="toast-actions">
      <button type="button" data-open-alert="${escapeHtml(key)}">Open incident</button>
    </div>
  `;
  stack.prepend(toast);
  while (stack.children.length > MAX_VISIBLE_TELEGRAM_TOASTS) {
    stack.lastElementChild?.remove();
  }
}

/**
 * Synchronize telegram popups with the current dashboard rows.
 */
function syncTelegramPopups(rows) {
  const telegramRows = (Array.isArray(rows) ? rows : []).filter(row =>
    row.telegram_sent && String(row.role || "").toLowerCase() === "primary"
  );
  const nowSeconds = Date.now() / 1000;
  const newRows = [];

  telegramRows.forEach(row => {
    const key = incidentKey(row);
    if (seenTelegramAlerts.has(key)) return;
    seenTelegramAlerts.add(key);
    const isFreshInitialAlert = !telegramAlertsInitialized
      && Number(row.epoch || 0) >= nowSeconds - INITIAL_ALERT_WINDOW_SECONDS;
    if (telegramAlertsInitialized || isFreshInitialAlert) {
      newRows.push(row);
    }
  });

  telegramAlertsInitialized = true;
  saveSeenTelegramAlerts();
  newRows.reverse().forEach(renderTelegramToast);
}

/**
 * Render chips into the dashboard DOM.
 */
function renderChips(items) {
  return (items || []).length
    ? `<div class="chip-row">${items.map(item => `<span class="chip">${escapeHtml(item)}</span>`).join("")}</div>`
    : `<p class="muted">No tactics recorded.</p>`;
}

/**
 * Render techniques into the dashboard DOM.
 */
function renderTechniques(items) {
  if (!items || !items.length) return `<p class="muted">No MITRE techniques recorded for this alert.</p>`;
  return `<div class="tech-list">${items.map(tech => `
    <article class="tech-card">
      <strong>${escapeHtml(tech.id)} - ${escapeHtml(tech.name)}</strong>
      ${tech.confidence ? `<small>Confidence: ${escapeHtml(tech.confidence)}</small>` : ""}
      ${tech.reason ? `<p>${escapeHtml(tech.reason)}</p>` : ""}
    </article>
  `).join("")}</div>`;
}

/**
 * Render actions into the dashboard DOM.
 */
function renderActions(items) {
  return `<ol class="action-list">${(items || []).map(action => `<li>${escapeHtml(action)}</li>`).join("")}</ol>`;
}

/**
 * Render secondary labels into the dashboard DOM.
 */
function renderSecondaryLabels(items) {
  if (!items || !items.length) return `<p class="muted">No secondary observations attached to this incident.</p>`;
  return `<div class="chip-row">${items.map(item =>
    `<span class="chip">${escapeHtml(item.name)} (${escapeHtml(item.value)})</span>`
  ).join("")}</div>`;
}

/**
 * Render sample flows into the dashboard DOM.
 */
function renderSampleFlows(items) {
  if (!items || !items.length) return `<p class="muted">No sample flows stored for this incident.</p>`;
  return `<div class="sample-flow-list">${items.map(flow => `
    <div class="sample-flow">
      <strong>${escapeHtml(flow.label)} - ${escapeHtml(flow.role)}</strong>
      <span>${escapeHtml(flow.proto || "-")} ${escapeHtml(flow.src || "-")}:${escapeHtml(flow.src_port || "-")} -> ${escapeHtml(flow.dst || "-")}:${escapeHtml(flow.dst_port || "-")}</span>
      <small>${escapeHtml(flow.time || "-")}</small>
    </div>
  `).join("")}</div>`;
}

/**
 * Render probabilities into the dashboard DOM.
 */
function renderProbabilities(items) {
  if (!items || !items.length) return `<p class="muted">No probability distribution stored.</p>`;
  const max = Math.max(0.0001, ...items.map(item => Number(item.value || 0)));
  return `<div class="prob-list">${items.map(item => `
    <div class="prob-row">
      <span>${escapeHtml(item.label)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(4, Number(item.value || 0) / max * 100)}%"></div></div>
      <strong>${Number(item.value || 0).toFixed(3)}</strong>
    </div>
  `).join("")}</div>`;
}

/**
 * Render block result into the dashboard DOM.
 */
function renderBlockResult(result) {
  if (!result) return `<p class="muted">No firewall action recorded.</p>`;
  const applied = result.applied ? "Applied" : "Not applied";
  const reason = result.skipped_reason ? ` (${result.skipped_reason})` : "";
  const ttl = result.ttl ? ` for ${result.ttl}s` : "";
  return `<p>${escapeHtml(applied + reason + ttl)}${result.ip ? ` - ${escapeHtml(result.ip)}` : ""}</p>`;
}

/**
 * Render reputation into the dashboard DOM.
 */
function renderReputation(reputation) {
  if (!reputation) return `<p class="muted">No reputation enrichment recorded.</p>`;
  const badge = reputation.badge || "Reputation available";
  const hits = reputation.hit_count ?? 0;
  return `<p>${escapeHtml(badge)} (${escapeHtml(hits)} hit(s))</p>`;
}

/**
 * Open incident details UI.
 */
function openIncidentDetails(index) {
  const row = latestIncidentRows[index];
  const modal = document.getElementById("incidentModal");
  const detail = document.getElementById("incidentDetail");
  if (!row || !modal || !detail) return;

  const flow = `${row.proto || "-"} ${row.src || "-"} -> ${row.dst || "-"}:${row.port || "-"}`;
  const incidentSummary = row.incident_summary || {};
  const routeDetail = row.router_confidence === null || row.router_confidence === undefined
    ? `${row.route || "-"}`
    : `${row.route || "-"} (${Number(row.router_confidence).toFixed(3)})`;

  detail.innerHTML = `
    <div class="detail-head">
      <div>
        <span class="eyebrow">${row.telegram_sent ? "Telegram-signaled threat" : "Logged threat"}</span>
        <h2 id="detailTitle">${escapeHtml(row.label)} attack detail</h2>
      </div>
      <span class="severity ${severityClass(row.severity)}">${escapeHtml(row.severity)}</span>
    </div>

    <div class="detail-grid">
      <section class="detail-section wide">
        <h3>Executive Summary</h3>
        <p>${escapeHtml(row.summary || "No summary recorded.")}</p>
        ${row.secondary_reason ? `<p class="muted">Secondary context: ${escapeHtml(row.secondary_reason)}</p>` : ""}
      </section>

      <section class="detail-section">
        <h3>Incident Scope</h3>
        <p>${escapeHtml(flow)}</p>
        <p class="muted">Time: ${escapeHtml(row.time)} - Primary: ${escapeHtml(row.primary_label || row.label)} - ${Number(row.flow_count || incidentSummary.flow_count || 1)} flow(s) - ${Number(row.possible_count || incidentSummary.possible_count || 0)} possible</p>
        ${renderSecondaryLabels(incidentSummary.secondary_labels)}
      </section>

      <section class="detail-section">
        <h3>Model Context</h3>
        <p>Route: ${escapeHtml(routeDetail)}</p>
        <p>Confidence: ${Number(row.confidence || 0).toFixed(3)}</p>
        ${row.llm_reclassified ? `<p class="muted">LLM reclassified a low-confidence/unknown flow.</p>` : ""}
        ${row.confidence_note ? `<p class="muted">${escapeHtml(row.confidence_note)}</p>` : ""}
      </section>

      <section class="detail-section wide">
        <h3>MITRE ATT&CK Mapping</h3>
        ${renderChips(row.mitre_tactics)}
        ${renderTechniques(row.mitre_techniques)}
      </section>

      <section class="detail-section wide">
        <h3>How to Respond / Actions to take</h3>
        ${renderActions(row.next_actions)}
      </section>

      <section class="detail-section wide">
        <h3>Sample Flows</h3>
        ${renderSampleFlows(row.sample_flows)}
      </section>

      <section class="detail-section">
        <h3>Firewall / Response</h3>
        ${renderBlockResult(row.block_result)}
      </section>

      <section class="detail-section">
        <h3>IP Reputation</h3>
        ${renderReputation(row.reputation)}
      </section>

      <section class="detail-section wide">
        <h3>Top Model Probabilities</h3>
        ${renderProbabilities(row.top_probabilities)}
      </section>
    </div>
  `;
  modal.hidden = false;
}

/**
 * Close incident details UI.
 */
function closeIncidentDetails() {
  const modal = document.getElementById("incidentModal");
  if (modal) modal.hidden = true;
}

/**
 * Load metrics from browser or API state.
 */
async function loadMetrics() {
  const response = await fetch("/api/metrics", { cache: "no-store" });
  if (!response.ok) throw new Error(`metrics failed: ${response.status}`);
  return response.json();
}

/**
 * Apply metrics to the current page.
 */
function applyMetrics(data) {
  const apiDot = document.getElementById("apiDot");
  apiDot?.classList.toggle("online", Boolean(data.api.online));
  setText("apiStatus", data.api.status);
  setText("modelName", data.api.model || "unavailable");
  setText("routingStatus", data.api.routing_enabled ? "auto" : "single");
  setText("updatedAt", data.generated_at);
  setText("pathText", data.paths.log_dir);

  setText("riskScore", data.kpis.risk_score);
  setText("threats24", fmt(data.kpis.threats_24h));
  setText("threats1h", possiblePhrase(data.kpis.possible_threats_24h));
  setText("highCritical", fmt(data.kpis.high_critical_24h));
  setText("analyzedRecords", fmt(data.kpis.analyzed_records_24h));
  setText("unknownRate", `${data.kpis.unknown_rate_24h}%`);
  setText("topSource", data.kpis.top_source || "-");
  setText("llmErrors", `${data.kpis.llm_errors_24h} LLM errors`);
  setText("routingText", data.api.routing_enabled ? "Router telemetry is enabled for model selection." : "Routing metadata unavailable.");

  renderSystemPulse(data.activity || {}, data.api || {});
  drawTimeline("timelineChart", data.series.timeline || []);
  drawDonut("labelChart", data.series.labels || []);
  drawDonut("routeChart", data.series.routes || [], "No route data");
  renderBars("severityBars", data.series.severities || []);
  renderRankList("portsList", data.series.ports || []);
  renderRankList("sourcesList", data.series.sources || []);
  renderRankList("destinationsList", data.series.destinations || []);
  renderIncidents(data.recent.attacks || []);
  syncTelegramPopups(latestIncidentRows);
}

/**
 * Initialize section nav behavior.
 */
function initSectionNav() {
  const links = Array.from(document.querySelectorAll("[data-section-link]"));
  if (!links.length) return;
  const sections = links
    .map(link => ({
      link,
      section: document.querySelector(link.getAttribute("href")),
    }))
    .filter(item => item.section);

  /**
   * Set active content.
   */
  function setActive(link) {
    links.forEach(item => item.classList.toggle("active", item === link));
  }

  /**
   * Update the active section link based on scroll position.
   */
  function updateActiveSection() {
    const offset = 140;
    let current = sections[0];
    for (const item of sections) {
      if (item.section.getBoundingClientRect().top <= offset) {
        current = item;
      }
    }
    if (current) setActive(current.link);
  }

  links.forEach(link => {
    link.addEventListener("click", () => setActive(link));
  });
  window.addEventListener("scroll", updateActiveSection, { passive: true });
  updateActiveSection();
}

/**
 * Refresh dashboard metrics from the API and update the page.
 */
async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    applyMetrics(await loadMetrics());
  } catch (err) {
    setText("apiStatus", "dashboard error");
    renderSystemPulse({
      state: "dead",
      label: "dead",
      headline: "Telemetry link interrupted",
      detail: "Dashboard could not load the metrics feed.",
    }, { online: false, status: "offline" });
    console.error(err);
  } finally {
    refreshInFlight = false;
  }
}

refresh();
initSectionNav();
initIncidentFilters();
setInterval(refresh, Math.max(2, refreshSeconds) * 1000);
document.getElementById("showMoreIncidents")?.addEventListener("click", () => {
  visibleIncidentCount += INCIDENT_PAGE_SIZE;
  renderIncidents(latestIncidentRows);
});
document.getElementById("incidentRows")?.addEventListener("click", event => {
  const button = event.target.closest("[data-incident-index]");
  if (!button) return;
  openIncidentDetails(Number(button.dataset.incidentIndex));
});
document.getElementById("telegramToastStack")?.addEventListener("click", event => {
  const dismissButton = event.target.closest("[data-dismiss-alert]");
  if (dismissButton) {
    dismissButton.closest(".telegram-toast")?.remove();
    return;
  }

  const openButton = event.target.closest("[data-open-alert]");
  if (!openButton) return;
  const key = openButton.dataset.openAlert;
  const index = latestIncidentRows.findIndex(row => incidentKey(row) === key);
  if (index >= 0) {
    openIncidentDetails(index);
    openButton.closest(".telegram-toast")?.remove();
  }
});
document.querySelectorAll("[data-close-details]").forEach(el => {
  el.addEventListener("click", closeIncidentDetails);
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape") closeIncidentDetails();
});
