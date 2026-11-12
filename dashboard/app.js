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

const DEFAULT_USER = "local-user";
const USERS_STORAGE_KEY = "llm_cli_users";
const USER_SCOPES_STORAGE_KEY = "llm_cli_user_scopes";

const readJson = (key, fallback) => {
  try {
    const value = JSON.parse(localStorage.getItem(key) || "null");
    return value ?? fallback;
  } catch {
    return fallback;
  }
};

const readUsers = () => {
  const saved = readJson(USERS_STORAGE_KEY, []);
  const current = localStorage.getItem("llm_cli_user_id") || DEFAULT_USER;
  const users = Array.isArray(saved)
    ? saved.filter((user) => typeof user === "string" && user.trim())
    : [];
  return [...new Set([DEFAULT_USER, ...users, current])];
};

const readUserScope = (userId) => {
  const scopes = readJson(USER_SCOPES_STORAGE_KEY, {});
  return scopes[userId] || {
    conversationId: makeId("conv"),
    sessionId: makeId("sess"),
  };
};

const state = {
  users: readUsers(),
  userId: localStorage.getItem("llm_cli_user_id") || DEFAULT_USER,
  conversationId: "",
  sessionId: "",
};

function saveUsers() {
  localStorage.setItem(USERS_STORAGE_KEY, JSON.stringify(state.users));
}

function saveUserScope() {
  const scopes = readJson(USER_SCOPES_STORAGE_KEY, {});
  scopes[state.userId] = {
    conversationId: state.conversationId,
    sessionId: state.sessionId,
  };
  localStorage.setItem(USER_SCOPES_STORAGE_KEY, JSON.stringify(scopes));
}

function persistScope() {
  state.userId = $("#user-id").value || DEFAULT_USER;
  state.conversationId = $("#conversation-id").value || makeId("conv");
  state.sessionId = $("#session-id").value || makeId("sess");
  if (!state.users.includes(state.userId)) state.users.push(state.userId);
  saveUsers();
  localStorage.setItem("llm_cli_user_id", state.userId);
  saveUserScope();
}

function renderUserOptions() {
  const select = $("#user-id");
  select.innerHTML = "";
  state.users.forEach((userId) => {
    const option = document.createElement("option");
    option.value = userId;
    option.textContent = userId;
    select.appendChild(option);
  });
  select.value = state.userId;
}

function loadScope() {
  renderUserOptions();
  const scope = readUserScope(state.userId);
  state.conversationId = scope.conversationId;
  state.sessionId = scope.sessionId;
  $("#conversation-id").value = state.conversationId;
  $("#session-id").value = state.sessionId;
  saveUsers();
  saveUserScope();
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
  setText(
    "#metric-route-reason",
    [route.reason, route.document_mode && route.document_mode !== "none" ? `documento: ${route.document_mode}` : ""]
      .filter(Boolean)
      .join(" · ") || "ruta seleccionada",
  );
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
    ["document", route.document_mode || "none"],
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
    const contentType = response.headers.get("content-type") || "";
    let payload;
    if (contentType.includes("application/json")) {
      payload = await response.json();
    } else {
      const rawText = await response.text();
      payload = { detail: rawText || `HTTP ${response.status}` };
    }
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

function switchUser(userId) {
  if (!userId || userId === state.userId) return;
  persistScope();
  state.userId = userId;
  localStorage.setItem("llm_cli_user_id", state.userId);
  const scope = readUserScope(state.userId);
  state.conversationId = scope.conversationId;
  state.sessionId = scope.sessionId;
  $("#conversation-id").value = state.conversationId;
  $("#session-id").value = state.sessionId;
  renderUserOptions();
  saveUserScope();
  resetTelemetry();
  refreshHistory();
}

function openNewUser() {
  const dialog = $("#new-user-dialog");
  $("#new-user-id").value = "";
  $("#new-user-id").setCustomValidity("");
  if (typeof dialog.showModal === "function") {
    dialog.showModal();
    $("#new-user-id").focus();
  } else {
    const userId = window.prompt("ID del nuevo usuario:");
    if (userId) createUser(userId);
  }
}

function createUser(userIdOrEvent) {
  if (typeof userIdOrEvent !== "string") userIdOrEvent.preventDefault();
  const input = $("#new-user-id");
  const userId = (typeof userIdOrEvent === "string" ? userIdOrEvent : input.value).trim();
  if (!userId || /[\r\n\t]/.test(userId)) {
    input.setCustomValidity("Escribe un ID válido.");
    input.reportValidity();
    return;
  }
  if (state.users.includes(userId)) {
    input.setCustomValidity("Ese usuario ya existe; selecciónalo desde el selector.");
    input.reportValidity();
    return;
  }
  input.setCustomValidity("");
  persistScope();
  state.users.push(userId);
  state.userId = userId;
  state.conversationId = makeId("conv");
  state.sessionId = makeId("sess");
  renderUserOptions();
  $("#conversation-id").value = state.conversationId;
  $("#session-id").value = state.sessionId;
  saveUsers();
  localStorage.setItem("llm_cli_user_id", state.userId);
  saveUserScope();
  const dialog = $("#new-user-dialog");
  if (dialog.open) dialog.close();
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
$("#user-id").addEventListener("change", (event) => switchUser(event.target.value));
$("#new-user").addEventListener("click", openNewUser);
$("#new-user-form").addEventListener("submit", createUser);
$("#cancel-new-user").addEventListener("click", () => $("#new-user-dialog").close());
