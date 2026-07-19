"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const RESOURCE_PROFILES = {
  eco: { cpu_percent: 35, gpu_memory_percent: 50, priority: "low" },
  balanced: { cpu_percent: 65, gpu_memory_percent: 75, priority: "normal" },
  performance: { cpu_percent: 100, gpu_memory_percent: 95, priority: "normal" },
};
const BROWSER_WORKER_PROFILES = {
  eco: { cpu_intensity: 25 },
  balanced: { cpu_intensity: 60 },
  performance: { cpu_intensity: 100 },
};
const state = {
  authenticated: false,
  episodes: [], assets: [], storage: [], reviewFaces: [], knownNames: [],
  selectedEpisode: "", selectedAsset: "", selectedStorage: "", selectedReview: "",
  pendingMutation: null, jsonRecord: "",
  browserWorker: null, browserWorkerActive: false, browserWorkerId: "", logicalCpuCount: 1,
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function formatDuration(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

function formatBytes(bytes) {
  let value = Number(bytes) || 0;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
  return `${value.toFixed(index ? 1 : 0)} ${units[index]}`;
}

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.className = "toast"; }, 3600);
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  if ((options.method || "GET") !== "GET") headers["X-Series-Web"] = "1";
  const response = await fetch(path, { ...options, headers, credentials: "same-origin", cache: "no-store" });
  let payload = {};
  try { payload = await response.json(); } catch (_) { payload = {}; }
  if (!response.ok) {
    if (response.status === 401) setAuthenticated(false);
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function setAuthenticated(authenticated) {
  state.authenticated = Boolean(authenticated);
  $("#adminArea").hidden = !state.authenticated;
  $("#adminState").hidden = !state.authenticated;
  $("#logoutButton").hidden = !state.authenticated;
  $("#loginButton").hidden = state.authenticated;
  $("#startBrowserWorkerButton").disabled = !state.authenticated || state.browserWorkerActive;
  if (!state.authenticated && state.browserWorker) stopBrowserWorker(false);
}

function renderPublic(payload) {
  const status = payload.status || {};
  const stats = payload.statistics || {};
  const percent = Math.max(0, Math.min(100, Number(status.progress_percent) || 0));
  $("#liveTitle").textContent = status.active ? (status.current_step || "Pipeline active") : "Pipeline idle";
  $("#liveDetail").textContent = status.active ? `${status.current_episode || "No episode ID"} · updated ${status.updated_age || "now"}` : `Last update ${status.updated_at_text || "not available"}`;
  $("#progressBar").style.width = `${percent}%`;
  $("#progressText").textContent = `${percent}%`;
  $("#etaText").textContent = status.eta_text || "calculating";
  $("#episodeCount").textContent = stats.generated_episode_count || 0;
  $("#finishedCount").textContent = stats.finished_episode_count || 0;
  $("#qualityAverage").textContent = `${Number(stats.average_quality_percent || 0).toFixed(0)}%`;
  $("#workerCount").textContent = status.active_worker_count || 0;
  $("#publicActivity").innerHTML = [
    ["Status", status.status || "UNKNOWN"], ["Episode", status.current_episode || "-"],
    ["Steps", `${status.completed_steps || 0}/${status.total_steps || 0}`], ["Updated", status.updated_age || "-"],
  ].map(([label, value]) => `<div><span class="muted">${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
  $("#publicSummary").innerHTML = [
    ["Release ready", stats.release_ready_count || 0], ["Season assets", `${stats.ready_season_asset_count || 0}/${stats.season_asset_count || 0}`],
    ["Regeneration queue", stats.open_regeneration_items || 0], ["Server time", payload.server_time || "-"],
  ].map(([label, value]) => `<div class="summary-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
  $("#connectionState").textContent = "NAS online";
  $("#connectionState").className = "connection-pill online";
  $("#lastRefresh").textContent = `Statistics refreshed: ${payload.server_time || "now"}`;
}

async function refreshPublic() {
  try { renderPublic(await api("/api/public/overview")); }
  catch (error) {
    $("#connectionState").textContent = "NAS unavailable";
    $("#connectionState").className = "connection-pill error";
  }
}

function renderEpisodes() {
  $("#episodeRows").innerHTML = state.episodes.map((row) => `<tr data-id="${escapeHtml(row.episode_id)}" class="${row.episode_id === state.selectedEpisode ? "selected" : ""}"><td><input type="checkbox" data-check="episodes" value="${escapeHtml(row.episode_id)}"></td><td><strong>${escapeHtml(row.display_title)}</strong><br><span class="muted">${escapeHtml(row.episode_id)}</span></td><td>${escapeHtml(row.readiness || "-")}</td><td>${Number(row.quality_percent || 0).toFixed(0)}%</td><td>${(Number(row.realism_score || 0) * 100).toFixed(0)}%</td><td>${row.scene_count || 0}</td><td>${escapeHtml(row.updated_at_text || "")}</td></tr>`).join("");
  $$("#episodeRows tr").forEach((row) => row.addEventListener("click", (event) => { if (event.target.matches("input")) return; state.selectedEpisode = row.dataset.id; renderEpisodes(); renderEpisodeDetails(); }));
}

function renderEpisodeDetails() {
  const row = state.episodes.find((item) => item.episode_id === state.selectedEpisode);
  if (!row) { $("#episodeDetails").textContent = "Select an episode."; return; }
  const issues = [...(row.blockers || []), ...(row.warnings || [])];
  $("#episodeDetails").innerHTML = `<strong>${escapeHtml(row.display_title)}</strong>${issues.length ? `<ul class="detail-list">${issues.slice(0, 12).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : '<p class="muted">No reported blockers or warnings.</p>'}`;
}

function renderAssets() {
  $("#assetRows").innerHTML = state.assets.map((row) => `<tr data-id="${escapeHtml(row.asset_id)}" class="${row.asset_id === state.selectedAsset ? "selected" : ""}"><td><input type="checkbox" data-check="assets" value="${escapeHtml(row.asset_id)}"></td><td><strong>${escapeHtml(row.display_title)}</strong></td><td>${escapeHtml(row.season_id)}</td><td>${escapeHtml(row.asset_kind)}</td><td>${escapeHtml(row.status)}</td><td>${formatDuration(row.duration_seconds)}</td><td>${escapeHtml(row.updated_at_text || "")}</td></tr>`).join("");
  $$("#assetRows tr").forEach((row) => row.addEventListener("click", (event) => { if (event.target.matches("input")) return; state.selectedAsset = row.dataset.id; renderAssets(); renderAssetDetails(); }));
}

function renderAssetDetails() {
  const row = state.assets.find((item) => item.asset_id === state.selectedAsset);
  if (!row) { $("#assetDetails").textContent = "Select an intro or outro."; return; }
  const issues = [...(row.blockers || []), ...(row.warnings || [])];
  $("#assetDetails").innerHTML = `<strong>${escapeHtml(row.display_title)}</strong> · ${formatDuration(row.duration_seconds)}${issues.length ? `<ul class="detail-list">${issues.slice(0, 12).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : '<p class="muted">No reported blockers or warnings.</p>'}`;
}

async function refreshAdmin() {
  const button = $("#refreshButton"); button.disabled = true;
  try {
    const payload = await api("/api/overview");
    state.episodes = payload.episodes || []; state.assets = payload.assets || [];
    renderPublic(payload); renderEpisodes(); renderAssets(); setAuthenticated(true);
    updateLocalControl(payload.status || {});
    maybeRestoreBrowserWorker();
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; }
}

function updateLocalControl(status) {
  const local = status.local_web_process || {};
  $("#stopPipelineButton").disabled = !local.active;
  $("#startPipelineButton").disabled = Boolean(status.active || local.active);
  const host = status.execution_host || "this web-server PC";
  $("#executionHost").textContent = host;
  const logicalCpus = Math.max(1, Number(status.logical_cpu_count) || 1);
  state.logicalCpuCount = logicalCpus;
  $("#cpuThreadHint").textContent = `${cpuThreads(logicalCpus)} of ${logicalCpus} logical CPU(s) will be available.`;
  const resources = local.resources || {};
  $("#runningResourceSummary").textContent = local.active
    ? `Running: ${resources.cpu_threads || "?"} CPU thread(s), ${resources.gpu_memory_percent ?? "?"}% GPU memory, ${resources.priority || "normal"} priority`
    : "No web-started pipeline";
  renderBrowserWorkerServerStatus(status.browser_workers || {});
}

function cpuThreads(logicalCpus) {
  const percent = Math.max(10, Math.min(100, Number($("#cpuPercent").value) || 65));
  return Math.max(1, Math.min(logicalCpus, Math.round(logicalCpus * percent / 100)));
}

function resourceSettings() {
  return {
    profile: $("#resourceProfile").value,
    cpu_percent: Number($("#cpuPercent").value),
    gpu_memory_percent: Number($("#gpuMemoryPercent").value),
    priority: $("#processPriority").value,
  };
}

function updateResourceLabels(markCustom = false) {
  if (markCustom) $("#resourceProfile").value = "custom";
  $("#cpuPercentValue").textContent = `${$("#cpuPercent").value}%`;
  $("#gpuMemoryPercentValue").textContent = `${$("#gpuMemoryPercent").value}%`;
  const logicalCpus = Math.max(1, Number(state.logicalCpuCount) || 1);
  $("#cpuThreadHint").textContent = `${cpuThreads(logicalCpus)} of ${logicalCpus} logical CPU(s) will be available.`;
}

function applyResourceProfile(profile) {
  const values = RESOURCE_PROFILES[profile];
  if (!values) return;
  $("#cpuPercent").value = values.cpu_percent;
  $("#gpuMemoryPercent").value = values.gpu_memory_percent;
  $("#processPriority").value = values.priority;
  updateResourceLabels(false);
}

function browserWorkerSettings() {
  return {
    profile: $("#browserWorkerProfile").value,
    cpu_intensity: Number($("#browserCpuIntensity").value) || 60,
  };
}

function updateBrowserWorkerLabels(markCustom = false) {
  if (markCustom) $("#browserWorkerProfile").value = "custom";
  $("#browserCpuValue").textContent = `${$("#browserCpuIntensity").value}%`;
  if (state.browserWorker) state.browserWorker.postMessage({ type: "configure", ...browserWorkerSettings() });
}

function applyBrowserWorkerProfile(profile) {
  const values = BROWSER_WORKER_PROFILES[profile];
  if (!values) return;
  $("#browserCpuIntensity").value = values.cpu_intensity;
  updateBrowserWorkerLabels(false);
}

function renderBrowserWorkerServerStatus(status) {
  const tasks = status.tasks || {};
  const activeCount = Number(status.active_count) || 0;
  if (!state.browserWorkerActive) {
    $("#browserWorkerDetail").textContent = activeCount
      ? `${activeCount} other browser worker(s) online · ${tasks.queued || 0} queued · ${tasks.leased || 0} active.`
      : `No browser worker online · ${tasks.queued || 0} frame check(s) queued.`;
  }
}

function setBrowserWorkerUi(active, label = "Offline") {
  state.browserWorkerActive = Boolean(active);
  $("#browserWorkerStatus").textContent = label;
  $("#browserWorkerStatus").className = `status-badge${active ? " pass" : ""}`;
  $("#startBrowserWorkerButton").disabled = active || !state.authenticated;
  $("#stopBrowserWorkerButton").disabled = !active;
  $("#queueBrowserChecksButton").disabled = !active;
}

function browserWorkerMessage(event) {
  const message = event.data || {};
  if (message.type === "registered") {
    state.browserWorkerId = message.worker_id || "";
    setBrowserWorkerUi(true, "Online");
    $("#browserWorkerDetail").textContent = "Connected to the NAS queue. Waiting for browser-compatible frame checks.";
  } else if (message.type === "task") {
    setBrowserWorkerUi(true, "Working");
    $("#browserWorkerDetail").textContent = `Analyzing a generated frame · ${message.completed || 0} completed · ${message.failed || 0} failed.`;
  } else if (message.type === "result") {
    setBrowserWorkerUi(true, "Online");
    $("#browserCompletedTasks").textContent = String(message.completed || 0);
    const score = Number(message.metric?.score || 0) * 100;
    $("#browserWorkerDetail").textContent = `Frame metric saved on the NAS (${score.toFixed(0)}%) · waiting for the next task.`;
  } else if (message.type === "idle") {
    setBrowserWorkerUi(true, "Online · idle");
    $("#browserCompletedTasks").textContent = String(message.completed || 0);
  } else if (message.type === "task-error" || message.type === "error") {
    $("#browserWorkerStatus").textContent = "Reconnect pending";
    $("#browserWorkerStatus").className = "status-badge fail";
    $("#browserWorkerDetail").textContent = message.error || "Browser worker connection failed.";
  } else if (message.type === "stopped") {
    if (state.browserWorker) state.browserWorker.terminate();
    state.browserWorker = null;
    state.browserWorkerId = "";
    setBrowserWorkerUi(false, "Offline");
  }
}

function startBrowserWorker(remember = true) {
  if (!state.authenticated) { toast("Administrator login is required for browser compute.", true); return; }
  if (!("Worker" in window) || !("OffscreenCanvas" in window) || !("createImageBitmap" in window)) {
    toast("This browser does not support the required Worker/Canvas APIs.", true);
    return;
  }
  if (state.browserWorker) return;
  state.browserWorker = new Worker("/browser-compute-worker.js?v=5");
  state.browserWorker.addEventListener("message", browserWorkerMessage);
  state.browserWorker.addEventListener("error", (event) => {
    $("#browserWorkerStatus").textContent = "Worker error";
    $("#browserWorkerStatus").className = "status-badge fail";
    $("#browserWorkerDetail").textContent = event.message || "Browser worker failed.";
  });
  setBrowserWorkerUi(true, "Connecting");
  state.browserWorker.postMessage({ type: "start", ...browserWorkerSettings() });
  if (remember) {
    try { localStorage.setItem("aiSeriesBrowserWorkerEnabled", "1"); } catch (_) { /* optional preference */ }
  }
}

function stopBrowserWorker(forget = true) {
  if (forget) {
    try { localStorage.removeItem("aiSeriesBrowserWorkerEnabled"); } catch (_) { /* optional preference */ }
  }
  if (!state.browserWorker) { setBrowserWorkerUi(false, "Offline"); return; }
  state.browserWorker.postMessage({ type: "stop" });
  const worker = state.browserWorker;
  setTimeout(() => {
    if (state.browserWorker === worker) {
      worker.terminate();
      state.browserWorker = null;
      state.browserWorkerId = "";
      setBrowserWorkerUi(false, "Offline");
    }
  }, 1600);
}

async function queueBrowserChecks() {
  const button = $("#queueBrowserChecksButton");
  button.disabled = true;
  try {
    const payload = await api("/api/browser-worker/queue", { method: "POST", body: JSON.stringify({ limit: 48 }) });
    toast(`${payload.queued || 0} recent frame check(s) queued.`);
    renderBrowserWorkerServerStatus(payload.status || {});
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = !state.browserWorkerActive; }
}

async function preparePwaCache() {
  $("#browserGpuStatus").textContent = navigator.gpu ? "Available" : "Unavailable";
  if (!("serviceWorker" in navigator)) { $("#pwaCacheStatus").textContent = "Unsupported"; return; }
  if (!window.isSecureContext) { $("#pwaCacheStatus").textContent = "HTTPS required"; return; }
  try {
    await navigator.serviceWorker.register("/service-worker.js", { scope: "/" });
    await navigator.serviceWorker.ready;
    $("#pwaCacheStatus").textContent = "Ready";
  } catch (error) {
    $("#pwaCacheStatus").textContent = "Failed";
  }
}

function maybeRestoreBrowserWorker() {
  if (!state.authenticated || state.browserWorker) return;
  try {
    if (localStorage.getItem("aiSeriesBrowserWorkerEnabled") === "1") startBrowserWorker(false);
  } catch (_) { /* optional preference */ }
}

function checkedIds(scope) { return $$(`input[data-check="${scope}"]:checked`).map((input) => input.value); }

function requestMutation(scope, action) {
  const ids = checkedIds(scope);
  if (!ids.length) { toast(`Check one or more ${scope} records first.`, true); return; }
  const confirmation = action === "delete" ? "DELETE" : "ARCHIVE";
  state.pendingMutation = { scope, action, ids, confirmation };
  $("#confirmTitle").textContent = `${action === "delete" ? "Delete" : "Archive"} ${ids.length} selected item(s)`;
  $("#confirmMessage").textContent = `Type ${confirmation} to continue. This explicitly changes NAS project data.`;
  $("#confirmationInput").value = ""; $("#confirmActionButton").textContent = action === "delete" ? "Delete permanently" : "Archive";
  $("#confirmDialog").showModal();
}

async function performMutation() {
  const item = state.pendingMutation;
  if (!item) return;
  const confirmation = $("#confirmationInput").value.trim().toUpperCase();
  if (confirmation !== item.confirmation) { toast(`Type ${item.confirmation} exactly.`, true); return; }
  try {
    await api(`/api/${item.scope}/mutate`, { method: "POST", body: JSON.stringify({ action: item.action, ids: item.ids, confirmation }) });
    $("#confirmDialog").close(); toast(`${item.action} completed.`); state.pendingMutation = null;
    if (item.scope === "storage") await loadStorage(); else await refreshAdmin();
  } catch (error) { toast(error.message, true); }
}

function playMedia(kind, id, title) {
  if (!id) { toast(`Select an ${kind} first.`, true); return; }
  $("#mediaTitle").textContent = title || id;
  $("#mediaPlayer").src = `/api/media?kind=${encodeURIComponent(kind)}&id=${encodeURIComponent(id)}`;
  $("#mediaDialog").showModal();
}

async function loadStorage() {
  const button = $("#loadStorageButton"); button.disabled = true; button.textContent = "Scanning NAS...";
  try { const payload = await api("/api/storage?limit=240"); state.storage = payload.records || []; renderStorage(); toast(`Loaded ${state.storage.length} records.`); }
  catch (error) { toast(error.message, true); }
  finally { button.disabled = false; button.textContent = "Scan project data"; }
}

function renderStorage() {
  $("#storageRows").innerHTML = state.storage.map((row) => `<tr data-id="${escapeHtml(row.record_id)}" class="${row.record_id === state.selectedStorage ? "selected" : ""}"><td><input type="checkbox" data-check="storage" value="${escapeHtml(row.record_id)}" ${row.archive_allowed || row.delete_allowed ? "" : "disabled"}></td><td>${escapeHtml(row.category)}</td><td><strong>${escapeHtml(row.relative_path)}</strong>${row.editable_json ? '<br><span class="status-badge pass">Editable JSON</span>' : ""}</td><td>${escapeHtml(row.kind)}</td><td>${formatBytes(row.size_bytes)}</td><td>${row.item_count || 0}</td><td>${escapeHtml(row.updated_at_text || "")}</td></tr>`).join("");
  $$("#storageRows tr").forEach((row) => row.addEventListener("click", (event) => { if (event.target.matches("input")) return; state.selectedStorage = row.dataset.id; renderStorage(); }));
}

async function editSelectedJson() {
  const record = state.storage.find((row) => row.record_id === state.selectedStorage);
  if (!record?.editable_json) { toast("Select an editable JSON database first.", true); return; }
  try { const payload = await api(`/api/storage/json?id=${encodeURIComponent(record.record_id)}`); state.jsonRecord = record.record_id; $("#jsonTitle").textContent = record.relative_path; $("#jsonEditor").value = payload.text; $("#jsonStatus").textContent = "A NAS-local backup is created before saving."; $("#jsonDialog").showModal(); }
  catch (error) { toast(error.message, true); }
}

async function saveJson() {
  try { JSON.parse($("#jsonEditor").value); } catch (error) { $("#jsonStatus").textContent = `Invalid JSON: ${error.message}`; return; }
  try { const payload = await api("/api/storage/json", { method: "POST", body: JSON.stringify({ id: state.jsonRecord, text: $("#jsonEditor").value }) }); $("#jsonStatus").textContent = `Saved. Backup: ${payload.result.backup_path}`; await loadStorage(); }
  catch (error) { $("#jsonStatus").textContent = error.message; }
}

async function loadReview() {
  const include = $("#includeNamed").checked ? "1" : "0";
  const button = $("#loadReviewButton"); button.disabled = true;
  try {
    const payload = await api(`/api/review?include_named=${include}`);
    state.reviewFaces = payload.faces || []; state.knownNames = payload.known_names || [];
    $("#knownNames").innerHTML = state.knownNames.map((name) => `<option value="${escapeHtml(name)}"></option>`).join("");
    const summary = payload.queue_summary || {};
    $("#reviewSummary").innerHTML = `<div class="summary-row"><span>Face cases shown</span><strong>${state.reviewFaces.length}</strong></div><div class="summary-row"><span>Open queue cases</span><strong>${summary.total || 0}</strong></div><div class="summary-row"><span>Voice clusters</span><strong>${(payload.voice_clusters || []).length}</strong></div>`;
    renderReview(); toast("Offline NAS review data loaded.");
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; }
}

function renderReview() {
  $("#reviewRows").innerHTML = state.reviewFaces.map((row) => `<tr data-id="${escapeHtml(row.cluster_id)}" class="${row.cluster_id === state.selectedReview ? "selected" : ""}"><td>${row.preview_available ? "●" : "–"}</td><td><strong>${escapeHtml(row.cluster_id)}</strong></td><td>${escapeHtml(row.name)}</td><td>${row.samples || 0}</td><td>${row.scenes || 0}</td><td>${escapeHtml(row.role_hint || "-")}</td></tr>`).join("");
  $$("#reviewRows tr").forEach((row) => row.addEventListener("click", () => { state.selectedReview = row.dataset.id; renderReview(); renderReviewEditor(); }));
}

function renderReviewEditor() {
  const row = state.reviewFaces.find((item) => item.cluster_id === state.selectedReview);
  if (!row) return;
  $("#reviewTitle").textContent = `${row.cluster_id} · ${row.samples || 0} samples`;
  $("#reviewName").value = row.name.startsWith("face_") ? "" : row.name;
  $("#reviewPriority").checked = Boolean(row.priority);
  $("#renameFrom").value = row.name.startsWith("face_") ? "" : row.name;
  if (row.preview_available) { $("#reviewPreview").src = `/api/review/preview?id=${encodeURIComponent(row.cluster_id)}&v=${Date.now()}`; $("#reviewPreview").hidden = false; $("#reviewPreviewEmpty").hidden = true; }
  else { $("#reviewPreview").removeAttribute("src"); $("#reviewPreview").hidden = true; $("#reviewPreviewEmpty").hidden = false; }
}

async function assignReview(nameOverride = "") {
  if (!state.selectedReview) { toast("Select a face cluster first.", true); return; }
  const name = nameOverride || $("#reviewName").value.trim();
  if (!name) { toast("Enter a character name.", true); return; }
  try { const payload = await api("/api/review/assign", { method: "POST", body: JSON.stringify({ cluster_id: state.selectedReview, name, priority: $("#reviewPriority").checked }) }); toast(`${state.selectedReview} saved as ${payload.result.name}.`); await loadReview(); }
  catch (error) { toast(error.message, true); }
}

async function renameReview() {
  const oldName = $("#renameFrom").value.trim(); const newName = $("#renameTo").value.trim();
  if (!oldName || !newName) { toast("Enter both current and corrected names.", true); return; }
  try { await api("/api/review/rename", { method: "POST", body: JSON.stringify({ old_name: oldName, new_name: newName, priority: $("#reviewPriority").checked }) }); toast(`${oldName} renamed to ${newName}.`); $("#renameTo").value = ""; await loadReview(); }
  catch (error) { toast(error.message, true); }
}

async function pipelineAction(action) {
  const resources = resourceSettings();
  const text = action === "start"
    ? `Start the offline full pipeline on ${$("#executionHost").textContent} with ${resources.cpu_percent}% CPU cores and ${resources.gpu_memory_percent}% GPU memory?`
    : "Stop only the pipeline process started by this web server on this PC?";
  if (!window.confirm(text)) return;
  const body = action === "start" ? JSON.stringify({ resources }) : "{}";
  try { const payload = await api(`/api/pipeline/${action}`, { method: "POST", body }); toast(action === "start" ? `Local pipeline started as PID ${payload.pid}.` : `Local pipeline PID ${payload.pid} stopped.`); await refreshAdmin(); await loadPipelineLog(); }
  catch (error) { toast(error.message, true); }
}

async function loadPipelineLog() { try { const payload = await api("/api/pipeline/log"); $("#pipelineLog").textContent = payload.text || "No log output."; $("#pipelineLog").scrollTop = $("#pipelineLog").scrollHeight; } catch (error) { toast(error.message, true); } }

function activateTab(name) {
  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${name}`));
  if (name === "logs") loadPipelineLog();
}

$("#loginButton").addEventListener("click", () => { $("#loginError").textContent = ""; $("#loginDialog").showModal(); });
$("#cancelLoginButton").addEventListener("click", () => $("#loginDialog").close());
$("#loginForm").addEventListener("submit", async (event) => { event.preventDefault(); try { await api("/api/auth/login", { method: "POST", body: JSON.stringify({ username: $("#usernameInput").value, password: $("#passwordInput").value }) }); $("#passwordInput").value = ""; $("#loginDialog").close(); setAuthenticated(true); await refreshAdmin(); } catch (error) { $("#loginError").textContent = error.message; } });
$("#logoutButton").addEventListener("click", async () => { try { await api("/api/auth/logout", { method: "POST", body: "{}" }); } catch (_) { /* session already gone */ } setAuthenticated(false); });
$("#refreshButton").addEventListener("click", refreshAdmin);
$("#startPipelineButton").addEventListener("click", () => pipelineAction("start"));
$("#stopPipelineButton").addEventListener("click", () => pipelineAction("stop"));
$("#resourceProfile").addEventListener("change", (event) => applyResourceProfile(event.target.value));
$("#cpuPercent").addEventListener("input", () => updateResourceLabels(true));
$("#gpuMemoryPercent").addEventListener("input", () => updateResourceLabels(true));
$("#processPriority").addEventListener("change", () => { $("#resourceProfile").value = "custom"; });
$("#browserWorkerProfile").addEventListener("change", (event) => applyBrowserWorkerProfile(event.target.value));
$("#browserCpuIntensity").addEventListener("input", () => updateBrowserWorkerLabels(true));
$("#startBrowserWorkerButton").addEventListener("click", () => startBrowserWorker(true));
$("#stopBrowserWorkerButton").addEventListener("click", () => stopBrowserWorker(true));
$("#queueBrowserChecksButton").addEventListener("click", queueBrowserChecks);
$("#loadStorageButton").addEventListener("click", loadStorage);
$("#editJsonButton").addEventListener("click", editSelectedJson);
$("#saveJsonButton").addEventListener("click", saveJson);
$("#loadReviewButton").addEventListener("click", loadReview);
$("#assignReviewButton").addEventListener("click", () => assignReview());
$("#statistReviewButton").addEventListener("click", () => assignReview("statist"));
$("#nofaceReviewButton").addEventListener("click", () => assignReview("noface"));
$("#renameReviewButton").addEventListener("click", renameReview);
$("#loadLogButton").addEventListener("click", loadPipelineLog);
$("#playEpisodeButton").addEventListener("click", () => { const row = state.episodes.find((item) => item.episode_id === state.selectedEpisode); playMedia("episode", state.selectedEpisode, row?.display_title); });
$("#playAssetButton").addEventListener("click", () => { const row = state.assets.find((item) => item.asset_id === state.selectedAsset); playMedia("asset", state.selectedAsset, row?.display_title); });
$("#closeMediaButton").addEventListener("click", () => { $("#mediaPlayer").pause(); $("#mediaPlayer").removeAttribute("src"); $("#mediaPlayer").load(); $("#mediaDialog").close(); });
$("#closeJsonButton").addEventListener("click", () => $("#jsonDialog").close());
$("#confirmActionButton").addEventListener("click", performMutation);
$$('[data-mutate]').forEach((button) => button.addEventListener("click", () => { const [scope, action] = button.dataset.mutate.split(":"); requestMutation(scope, action); }));
$$(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));

setAuthenticated(false);
applyResourceProfile("balanced");
applyBrowserWorkerProfile("balanced");
setBrowserWorkerUi(false, "Offline");
preparePwaCache();
refreshPublic();
setInterval(refreshPublic, 10_000);
api("/api/health").then((payload) => { if (payload.authenticated) { setAuthenticated(true); refreshAdmin(); } }).catch(() => {});
