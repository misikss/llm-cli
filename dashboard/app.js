const $ = (selector) => document.querySelector(selector);

const makeId = (prefix) => {
  const value = window.crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
  return `${prefix}_${value.slice(0, 12)}`;
};

const formatNumber = (value) => new Intl.NumberFormat("en-US").format(Number(value || 0));
const formatCost = (value) => `$${Number(value || 0).toFixed(8)}`;
const formatLatency = (value) => `${Number(value || 0).toFixed(2)} ms`;

const setText = (selector, value) => {
  const element = $(selector);
  if (element) element.textContent = value ?? "—";
};

const state = {
  userId: localStorage.getItem("llm_cli_user_id") || "local-user",
  conversationId: localStorage.getItem("llm_cli_conversation_id") || makeId("conv"),
  sessionId: sessionStorage.getItem("llm_cli_session_id") || makeId("sess"),
};

function persistScope() {
  state.userId = $("#user-id").value.trim() || "local-user";
  state.conversationId = $("#conversation-id").value.trim() || "default-conversation";
  state.sessionId = $("#session-id").value.trim() || "dashboard-session";
  localStorage.setItem("llm_cli_user_id", state.userId);
  localStorage.setItem("llm_cli_conversation_id", state.conversationId);
  sessionStorage.setItem("llm_cli_session_id", state.sessionId);
}

function loadScope() {
  $("#user-id").value = state.userId;
  $("#conversation-id").value = state.conversationId;
  $("#session-id").value = state.sessionId;
}

function resetTelemetry() {
  setText("#metric-route", "—");
  setText("#metric-route-reason", "waiting for run");
  setText("#metric-tokens", "—");
  setText("#metric-token-scope", "generation only");
  setText("#metric-latency", "—");
  setText("#metric-cost", "—");
  setText("#metric-model", "waiting for model");
  setText("#trace-pill", "NO TRACE");
  $("#trace-pill").classList.remove("ready");
  $("#empty-state").hidden = false;
  $("#response-content").hidden = true;
  $("#stage-body").innerHTML = '<tr class="table-empty"><td colspan="5">No stages recorded yet.</td></tr>';
  $("#raw-json").textContent = "Waiting for a run...";
}

function renderFilePreview(file) {
  const preview = $("#file-preview");
  if (!file) {
    preview.hidden = true;
    preview.textContent = "";
    return;
  }
  preview.hidden = false;
  preview.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
}

function renderResponse(payload) {
  const response = payload.response || {};
  const route = payload.route_decision || {};
  const metrics = payload.metrics || {};
  const request = payload.request || {};

  setText("#metric-route", route.route || metrics.routed_as || "—");
  setText("#metric-route-reason", route.reason || "route selected");
  setText("#metric-tokens", formatNumber(metrics.tokens_totales));
  setText("#metric-token-scope", metrics.token_usage_scope || "pipeline");
  setText("#metric-latency", formatLatency(metrics.total_latency_ms));
  setText("#metric-cost", formatCost(metrics.costo_total_usd));
  setText("#metric-model", metrics.model || "unknown model");
  setText("#trace-pill", metrics.trace_id || "NO TRACE");
  $("#trace-pill").classList.add("ready");

  $("#empty-state").hidden = true;
  $("#response-content").hidden = false;

  const meta = $("#response-meta");
  meta.innerHTML = "";
  [
    ["category", response.category],
    ["confidence", `${Math.round(Number(response.confidence || 0) * 100)}%`],
    ["user", request.user_id],
    ["conversation", request.conversation_id],
    ["session", request.session_id],
  ].forEach(([label, value]) => {
    const chip = document.createElement("span");
    chip.textContent = `${label}: ${value || "—"}`;
    meta.appendChild(chip);
  });

  setText("#answer-text", response.answer || "No answer returned.");
  const points = $("#key-points");
  points.innerHTML = "";
  (response.key_points || ["No key points returned."]).forEach((point) => {
    const item = document.createElement("li");
    item.textContent = point;
    points.appendChild(item);
  });

  const sources = $("#sources");
  sources.innerHTML = "";
  (response.sources_suggested || ["No sources returned."]).forEach((source) => {
    const item = document.createElement("li");
    item.textContent = source;
    sources.appendChild(item);
  });

  const stages = $("#stage-body");
  stages.innerHTML = "";
  if (!metrics.stage_metrics?.length) {
    stages.innerHTML = '<tr class="table-empty"><td colspan="5">No stages recorded.</td></tr>';
  } else {
    metrics.stage_metrics.forEach((stage) => {
      const row = document.createElement("tr");
      const metadata = stage.metadata && Object.keys(stage.metadata).length
        ? JSON.stringify(stage.metadata)
        : "—";
      row.innerHTML = `
        <td>${stage.name || "—"}</td>
        <td class="status-${stage.status || "ok"}">${stage.status || "ok"}</td>
        <td>${formatLatency(stage.latency_ms)}</td>
        <td>${formatNumber(stage.input_tokens)} / ${formatNumber(stage.output_tokens)}</td>
        <td class="metadata-cell"></td>
      `;
      row.querySelector(".metadata-cell").textContent = metadata;
      stages.appendChild(row);
    });
  }

  $("#raw-json").textContent = JSON.stringify(payload, null, 2);
}

function renderError(message) {
  $("#empty-state").hidden = true;
  $("#response-content").hidden = false;
  $("#response-meta").innerHTML = '<span class="error-chip">request error</span>';
  $("#answer-text").textContent = message;
  $("#key-points").innerHTML = "";
  $("#sources").innerHTML = "";
  $("#stage-body").innerHTML = `<tr><td colspan="5" class="status-error"></td></tr>`;
  $("#stage-body td").textContent = message;
  $("#raw-json").textContent = JSON.stringify({ error: message }, null, 2);
}

async function runHealthCheck() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    $("#health-dot").classList.toggle("online", health.ok && health.configured);
    $("#health-dot").classList.toggle("offline", !health.ok || !health.configured);
    setText("#health-label", health.configured ? "LOCAL API / READY" : "LOCAL API / MISSING KEY");
  } catch {
    $("#health-dot").classList.add("offline");
    setText("#health-label", "LOCAL API / OFFLINE");
  }
}

async function refreshHistory() {
  persistScope();
  const params = new URLSearchParams({
    limit: "12",
    user_id: state.userId,
    conversation_id: state.conversationId,
  });
  try {
    const response = await fetch(`/api/history?${params}`);
    const data = await response.json();
    const list = $("#history-list");
    list.innerHTML = "";
    if (!data.entries?.length) {
      list.innerHTML = '<p class="table-empty">No runs in this conversation yet.</p>';
      return;
    }
    [...data.entries].reverse().forEach((entry) => {
      const item = document.createElement("article");
      item.className = "history-item";
      const question = document.createElement("p");
      question.className = "history-question";
      question.textContent = entry.question || "Untitled request";
      const details = document.createElement("div");
      details.className = "history-details";
      details.innerHTML = `<span>${entry.routed_as || "general"} · ${formatNumber(entry.tokens_totales)} tok</span><span>${formatCost(entry.costo_total_usd)}</span>`;
      item.append(question, details);
      list.appendChild(item);
    });
  } catch {
    $("#history-list").innerHTML = '<p class="table-empty">History unavailable.</p>';
  }
}

async function submitRun(event) {
  event.preventDefault();
  persistScope();

  const form = $("#ask-form");
  const button = $("#submit-button");
  const label = $("#submit-label");
  const file = $("#file").files[0];
  const data = new FormData();
  data.append("question", $("#question").value.trim());
  data.append("user_id", state.userId);
  data.append("session_id", state.sessionId);
  data.append("conversation_id", state.conversationId);
  data.append("use_grounding", $("#use-grounding").checked ? "true" : "false");
  data.append("use_weather", $("#use-weather").checked ? "true" : "false");
  if (file) data.append("file", file);

  button.disabled = true;
  label.textContent = "Running pipeline...";
  $("#trace-pill").textContent = "RUNNING";
  try {
    const response = await fetch("/api/ask", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Request failed.");
    renderResponse(payload);
    await refreshHistory();
  } catch (error) {
    renderError(error.message);
  } finally {
    button.disabled = false;
    label.textContent = "Run pipeline";
  }
}

function setupUpload() {
  const dropzone = $("#dropzone");
  const input = $("#file");
  input.addEventListener("change", () => renderFilePreview(input.files[0]));
  ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  }));
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (!file) return;
    input.files = event.dataTransfer.files;
    renderFilePreview(file);
  });
}

function newConversation() {
  const conversationId = makeId("conv");
  $("#conversation-id").value = conversationId;
  persistScope();
  resetTelemetry();
  refreshHistory();
}

loadScope();
setupUpload();
resetTelemetry();
runHealthCheck();
refreshHistory();
$("#ask-form").addEventListener("submit", submitRun);
$("#refresh-history").addEventListener("click", refreshHistory);
$("#new-conversation").addEventListener("click", newConversation);
["#user-id", "#conversation-id", "#session-id"].forEach((selector) => $(selector).addEventListener("change", () => {
  persistScope();
  refreshHistory();
}));
