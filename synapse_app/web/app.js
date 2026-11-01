"use strict";

import { ChatStream } from "./chat-stream.mjs";
import { navigate } from "./navigation.mjs";
import { MonitorView } from "./monitor.mjs";
import { UpdateView } from "./updates.mjs";
import { bindChatInput } from "./chat-input.mjs";
import { bindReasoningEffort, draftControls, elapsed, loadingText, modelLoadPayload, ModelPrecheck } from "./model-controls.mjs";

const el = (id) => document.getElementById(id);
let status = null;
let controller = null;
let actionPending = false;
let history = [];
let settingsInitialized = false;
let browserLocation = null;
let browserBusy = false;
let appliedTuningAt = null;
const modelPrecheck = new ModelPrecheck({
  input: el("model"), button: el("precheck-model"), output: el("model-precheck"), request, changed: controls,
});

function notice(message = "") {
  el("notice").textContent = message;
  el("notice").hidden = !message;
  el("controls-notice").textContent = message;
  el("controls-notice").hidden = !message;
}

async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || `Request failed (${response.status}).`);
  return data;
}

function controls() {
  const tuning = !!status?.tuning?.active;
  const active = tuning || (status && ["loading", "ready", "stopping"].includes(status.state));
  el("load").disabled = !status?.backend_available || active || actionPending || !!controller;
  el("unload").disabled = !active || tuning || status?.state === "stopping" || actionPending;
  el("send").disabled = status?.state !== "ready" || tuning || !!controller || actionPending || status?.chat_busy;
  el("stop").disabled = !controller;
  el("clear").disabled = !!controller;
  el("max-output").disabled = !!controller;
  el("reasoning-effort").disabled = !!controller;
  el("quick-reasoning-effort").disabled = !!controller;
  el("start-tuning").disabled = tuning || !!controller || actionPending || status?.chat_busy ||
    !status?.backend_available || ["loading", "stopping"].includes(status?.state) || !el("tuning-consent").checked;
  el("cancel-tuning").disabled = !tuning || actionPending;
  el("cancel-work").disabled = actionPending || status?.state === "stopping";
  el("tuning-goal").disabled = tuning || actionPending;
  el("tuning-hybrid").disabled = tuning || actionPending || !status?.hybrid_available;
  el("tuning-consent").disabled = tuning || actionPending;
  el("browse-models").disabled = active || actionPending || browserBusy;
  el("precheck-model").disabled = !status || actionPending || modelPrecheck.busy || !el("model").value.trim();
  for (const button of el("model-browser").querySelectorAll("button")) {
    button.disabled = active || actionPending || browserBusy;
  }
  el("browser-path").disabled = active || actionPending || browserBusy;
  if (browserLocation && !browserLocation.parent) el("browser-up").disabled = true;
  for (const id of ["model", "context", "placement", "kv-cache-type", "cpu-threads", "batch-threads", "cpu-poll", "host-op-offload", "op-offload-min-batch", "load-mode", "expert-cache", "fit-target", "draft-type", "draft-model", "draft-tokens"]) {
    el(id).disabled = active || actionPending;
  }
  el("execution-mode").disabled = active || actionPending;
  const selectedMode = el("execution-mode").value;
  if (selectedMode === "auto") {
    for (const id of ["context", "placement", "cpu-threads", "batch-threads", "cpu-poll", "host-op-offload", "op-offload-min-batch", "load-mode", "expert-cache", "fit-target", "draft-model", "draft-tokens"]) {
      el(id).disabled = true;
    }
  }
  const draft = draftControls(status, selectedMode, el("draft-type").value);
  el("draft-type").disabled = active || actionPending || draft.disabled;
  el("draft-type").querySelector('option[value="dflash"]').disabled = draft.dflashDisabled;
  el("draft-type").querySelector('option[value="mtp"]').disabled = draft.mtpDisabled;
  for (const id of ["draft-model", "draft-tokens"]) el(id).disabled = active || actionPending || draft.inputDisabled;
  el("draft-tokens").max = draft.maximum;
  if (status?.backend_kind !== "llama.cpp") {
    el("kv-cache-type").disabled = true;
    el("cpu-poll").disabled = true;
    el("host-op-offload").disabled = true;
    el("op-offload-min-batch").disabled = true;
  }
  if (selectedMode === "hybrid") {
    for (const id of ["placement", "context", "load-mode", "fit-target", "host-op-offload", "op-offload-min-batch"]) {
      el(id).disabled = true;
    }
    el("expert-cache").disabled = active || actionPending || !status?.hybrid_generic;
  }
  el("placement").querySelector('option[value="cpu"]').disabled =
    status?.backend_kind === "ik_llama.cpp" || selectedMode !== "standard";
}

function renderStatus() {
  const labels = { stopped: "Stopped", loading: "Loading model", ready: "Ready", stopping: "Unloading", error: "Backend error" };
  const experimental = status.backend_kind === "ik_llama.cpp";
  const hybrid = (status.execution_profile || "standard").startsWith("experimental-hybrid-");
  const transfer = status.hybrid_transfer === "staged" ? "bounded staging" : "direct DMA";
  const hybridContext = status.hybrid_context ?? status.context;
  const hybridBudget = status.expert_cache_mib ?? status.hybrid_cache_mib;
  const pipeline = status.hybrid_decode_pipeline
    ? ` · Decode pipeline: overlap ${status.hybrid_overlap_verified ? "confirmed" : "awaiting eligible decode"}; gate/up fusion ${status.hybrid_gate_up_verified ? "confirmed" : "not yet observed"}${status.hybrid_async_ids ? `; event-protected GPU IDs ${status.hybrid_async_ids_verified ? "configured" : "awaiting backend"}` : ""}`
    : "";
  el("execution-mode").querySelector('option[value="hybrid"]').disabled = !status.hybrid_available;
  el("selection-reason").textContent = status.selection_reason || "";
  el("profile-note").textContent = status.hybrid_generic
    ? "Automatic detects routed-MoE dimensions, validates plan bindings, and prefers native fitting for small weight files. Unbound layouts use a causal plan with their configured budget and transfer policy. Backend operations and coverage must verify before Ready; speedups are not universal."
    : status.hybrid_available ? "This installation uses a model-bound or explicitly supplied plan. A version-2 installation enables generic automatic fallback; other models use native execution."
    : "No hybrid installation profile is configured. Automatic uses the standard backend. Select Standard for manual settings.";
  el("execution-warning").hidden = !status.execution_warning;
  el("execution-warning").textContent = status.execution_warning || "";
  el("execution-footer").textContent = hybrid
    ? `Experimental engine - Hybrid CPU/GPU dispatch - Adaptive ${transfer}${pipeline} - Not quality-qualified`
    : `Release candidate - ${experimental ? "Experimental ik_llama.cpp" : "Standard llama.cpp"} execution - Local only`;
  el("placement-note").textContent = hybrid
    ? `Experimental MoE profile: ${hybridContext.toLocaleString()} context and ${hybridBudget.toLocaleString()} MiB resident expert bank from the installation policy. Fitting, repacking, and host-op offload are disabled. Unload releases resources.`
    : "Automatic uses llama.cpp fitting with the selected VRAM margin. It is not a RAM-fit guarantee. Larger contexts need more memory; 256K requires model and backend support.";
  el("cache-note").textContent = hybrid
    ? `Residency is derived from routed expert tensors, not a model-name allowlist. Cold experts execute on CPU; causal admissions use ${transfer}. Original weights, biases, activations and router probabilities are retained. New chat does not reset residency; unload/reload does.`
    : "Fewer threads can be faster. RAM loading can increase memory pressure. The upstream experimental expert cache uses VRAM and may be slower. Zero disables it.";
  if (status.hybrid_generic) el("cache-note").textContent +=
    " Require hybrid mode allows a smaller manual cache for unbound causal layouts. Bound training plans retain their configured budget. Automatic defaults are unchanged.";
  el("speculation-note").textContent = status.speculation_enabled
    ? "Experimental speculation is enabled. Native MTP supports Standard and Hybrid with a compatible backend; DFlash is Standard only. Extra VRAM and a matching draft are required. Hybrid MTP may need a smaller expert bank. Speed and quality are not guaranteed."
    : "Speculation is disabled by default. Research trials require --experimental-speculation at launch.";
  el("cpu-threads").max = status.logical_cpus;
  el("batch-threads").max = status.logical_cpus;
  el("placement").querySelector('option[value="cpu"]').disabled = experimental || hybrid;
  el("placement").querySelector('option[value="auto"]').textContent = hybrid
    ? `Experimental hybrid CPU / GPU (${transfer})` : "Automatic CPU / GPU placement";
  el("expert-cache").max = experimental ? "0" : "65536";
  el("state").textContent = status.tuning?.active ? "Auto tuning" : labels[status.state] || status.state;
  el("state").dataset.state = status.tuning?.active ? "loading" : status.state;
  updateView.setVersion(status.app_version);
  el("sidebar-model").textContent = status.model ? status.model.split(/[\\/]/).at(-1) : "No model loaded";
  el("connection-state").textContent = status.tuning?.active ? "Auto Tune owns this model. Cancel it or wait before connecting apps."
    : status.state === "ready" ? "Ready. Your apps can connect now."
    : "No model is ready. Open Chat → Model Controls to load a model before sending API requests.";
  el("controls-action-status").textContent = labels[status.state] || status.state;
  el("model-status").textContent = status.model || "No model loaded.";
  el("run-status").textContent = status.model
    ? `${status.backend_kind}${experimental || hybrid ? " (experimental)" : ""} · ${hybrid ? `Hybrid ${transfer}${status.hybrid_verified ? " (activation verified)" : " (verifying activation)"}` : status.placement === "cpu" ? "CPU only" : "Automatic placement"} · ${status.context.toLocaleString()} context · ${status.cpu_threads} generation / ${status.batch_threads} prompt threads · ${status.load_mode} loading · ${status.expert_cache_mib} MiB ${hybrid ? "resident expert bank · fitting off" : `expert cache · ${status.fit_target_mib} MiB reserve`} · polling ${status.cpu_poll} · host-op offload ${status.host_op_offload ? "on" : "off"} (min batch ${status.op_offload_min_batch})${status.draft_model ? ` · ${status.draft_type === "mtp" ? "MTP" : "DFlash"} (up to ${status.draft_tokens} draft tokens${status.speculation_verified ? ", activation verified" : ", activation unverified"})` : ""}`
    : "Load an existing GGUF to begin.";
  if (status.model) el("run-status").textContent += ` · Attention KV ${status.kv_cache_type || "f16"}`
    + (status.kv_cache_verified ? " (backend confirmed)" : " (backend confirmation unavailable)");
  el("logs").textContent = status.logs.join("\n") || "No output yet.";
  const resources = status.resources;
  if (status.resource_error) {
    el("resources").textContent = status.resource_error;
  } else if (resources) {
    const gib = (bytes) => typeof bytes === "number" ? (bytes / 1024 ** 3).toFixed(1) : "unavailable";
    const cpu = resources.cpu_logical_cores === null ? "sampling"
      : `${resources.cpu_logical_cores.toFixed(1)} logical cores (${(100 * resources.cpu_logical_cores / status.logical_cpus).toFixed(0)}% of machine)`;
    const breakdown = [["file-backed", resources.file_backed_bytes], ["anonymous", resources.anonymous_bytes],
      ["shared/driver-mapped", resources.shared_bytes]]
      .filter(([, bytes]) => typeof bytes === "number").map(([name, bytes]) => `${gib(bytes)} GiB ${name}`).join(", ");
    el("resources").textContent = `CPU: ${cpu}. Resident RAM: ${gib(resources.rss_bytes)} GiB `
      + `(${breakdown || "breakdown unavailable on this platform"}). `
      + `System available: ${gib(resources.system_available_ram_bytes)} GiB, including reclaimable cache.`;
  } else {
    el("resources").textContent = status.model ? "Collecting process resource usage..." : "";
  }
  if (status.model && ["loading", "ready", "stopping"].includes(status.state)) {
    if (el("model").value !== status.model) modelPrecheck.invalidate();
    el("model").value = status.model;
    el("context").value = status.context;
    el("placement").value = status.placement;
  }
  if (!settingsInitialized || (status.model && ["loading", "ready", "stopping"].includes(status.state))) {
    el("execution-mode").value = status.execution_mode || "standard";
    if (hybrid) {
      el("context").value = status.context;
      el("placement").value = status.placement;
    }
    el("cpu-threads").value = status.cpu_threads;
    el("batch-threads").value = status.batch_threads;
    el("cpu-poll").value = status.cpu_poll;
    el("host-op-offload").value = String(status.host_op_offload);
    el("op-offload-min-batch").value = status.op_offload_min_batch;
    el("kv-cache-type").value = status.kv_cache_type || "f16";
    el("load-mode").value = status.load_mode;
    el("expert-cache").value = status.expert_cache_mib;
    el("fit-target").value = status.fit_target_mib;
    el("draft-model").value = status.draft_model || "";
    el("draft-type").value = status.draft_type || (status.draft_model ? "dflash" : "none");
    el("draft-tokens").value = status.draft_tokens;
    settingsInitialized = true;
  }
  el("backend-help").hidden = status.backend_available;
  el("backend-help").textContent = "Backend missing. Install llama.cpp, then start SYNAPSE with: synapse --llama-server /path/to/llama-server (llama-server.exe on Windows). No backend or model is downloaded automatically.";
  if (status.error) notice(status.error);
  else if (status.context_warning) notice(status.context_warning);
  contextNote();
  renderTuning();
  controls();
}

function renderTuning() {
  const tuning = status.tuning;
  const working = status.state === "loading" || tuning?.active;
  el("model-work").hidden = !working;
  if (working) {
    el("work-title").textContent = tuning?.active ? "Auto tuning your model" : "Loading model";
    el("work-detail").textContent = tuning?.active
      ? `${tuning.phase} · ${elapsed(tuning.elapsed_seconds)} total. ${loadingText(status.loading)}`
      : loadingText(status.loading);
    el("work-output").textContent = tuning?.active && !status.loading
      ? tuning.detail || "Local benchmarks can take time. You can read this page or cancel and restore."
      : status.loading?.latest_output || "Waiting for the first backend message. You can cancel loading.";
    el("cancel-work").textContent = tuning?.active ? "Cancel Auto Tune" : "Cancel loading";
    if (!status.loading && typeof tuning?.progress === "number") {
      el("work-progress").max = 1;
      el("work-progress").value = tuning.progress;
    } else {
      el("work-progress").removeAttribute("value");
    }
  }
  if (!tuning) return;
  el("tuning-status").textContent = tuning.state === "idle" ? "No tuning run yet."
    : `${tuning.phase || tuning.state} · ${elapsed(tuning.elapsed_seconds)}. ${tuning.scope || ""}`;
  el("tuning-error").hidden = !tuning.error;
  el("tuning-error").textContent = tuning.error || "";
  const trials = tuning.trials || [];
  el("tuning-trials").hidden = !trials.length;
  el("tuning-trial-list").replaceChildren(...trials.map((trial) => {
    const item = document.createElement("li");
    const settings = trial.settings;
    item.textContent = `${settings.execution_mode}, ${settings.context.toLocaleString()} context`
      + (settings.expert_cache_mib ? `, ${settings.expert_cache_mib} MiB expert bank` : "")
      + (trial.capacity_passed ? `: passed ${trial.prompt_tokens.toLocaleString()} input tokens in ${elapsed(trial.request_seconds)}.`
        : `: failed - ${trial.error || "no completed capacity check"}`);
    return item;
  }));
  const recommendation = tuning.recommendation;
  el("tuning-result").hidden = !recommendation;
  if (!recommendation) return;
  const settings = recommendation.settings;
  const apps = recommendation.apps;
  const current = !tuning.active && status.state === "ready" && status.model === tuning.model &&
    Object.entries(settings).every(([key, value]) => status[key] === value) &&
    (settings.kv_cache_type || "f16") === (status.kv_cache_type || "f16");
  if (current && appliedTuningAt !== tuning.completed_at) {
    el("max-output").value = apps.maxOutputTokens;
    appliedTuningAt = tuning.completed_at;
    contextNote();
  }
  el("tuning-summary").textContent = `${current ? "Applied" : "Previous result; not currently applied"}: `
    + `${settings.context.toLocaleString()} context, ${settings.execution_mode}. `
    + `Verified ${recommendation.prompt_tokens_tested.toLocaleString()} input tokens and a short anchor-retrieval response. `
    + `Context ${apps.context.toLocaleString()}, response allowance ${apps.maxOutputTokens.toLocaleString()}, `
    + `safety margin ${apps.safetyMarginTokens.toLocaleString()} tokens. Model ID: local-model.`;
  el("tuning-app-settings").textContent = JSON.stringify({
    maxInputTokens: apps.maxInputTokens, maxOutputTokens: apps.maxOutputTokens,
  }, null, 2);
  el("copy-tuning-settings").disabled = !current;
}

function contextNote() {
  el("context-note").textContent = status?.state === "ready" && Number(el("max-output").value) > status.context
    ? `Loaded context: ${status.context.toLocaleString()} tokens. This may stop generation before the output budget is reached. Unload and reload with a larger context in Run settings below.`
    : "";
}

el("max-output").addEventListener("input", contextNote);
bindReasoningEffort(el("reasoning-effort"), el("quick-reasoning-effort"), notice);
el("tuning-consent").addEventListener("change", controls);
el("start-tuning").addEventListener("click", async () => {
  actionPending = true;
  notice();
  controls();
  try {
    status = await request("/api/v1/tuning/start", {
      model: el("model").value, goal: el("tuning-goal").value, placement: el("placement").value,
      kv_cache_type: el("kv-cache-type").value,
      include_hybrid: el("tuning-hybrid").checked, confirm: el("tuning-consent").checked,
    });
    renderStatus();
  } catch (error) {
    notice(error.message);
  } finally {
    actionPending = false;
    controls();
  }
});
async function cancelTuning() {
  actionPending = true;
  controls();
  try {
    status = await request("/api/v1/tuning/cancel", {});
    renderStatus();
  } catch (error) {
    notice(error.message);
  } finally {
    actionPending = false;
    controls();
  }
}
el("cancel-tuning").addEventListener("click", cancelTuning);
el("cancel-work").addEventListener("click", () => {
  if (status.tuning?.active) cancelTuning();
  else el("unload").click();
});
el("copy-tuning-settings").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(el("tuning-app-settings").textContent);
    notice("VS Code limits copied. Update your app's model configuration before reconnecting.");
  } catch (error) {
    notice(`Clipboard unavailable. Copy the displayed limits manually. ${error.message}`);
  }
});
el("open-controls").addEventListener("click", () => el("model-controls").showModal());
el("close-controls").addEventListener("click", () => el("model-controls").close());
el("model-controls").addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    event.preventDefault();
    el("model-controls").close();
  }
});

async function browseModels(path) {
  if (browserBusy || el("browse-models").disabled) return;
  browserBusy = true;
  el("model-browser").hidden = false;
  el("browser-status").textContent = "Reading local folder...";
  el("browser-error").hidden = true;
  el("browser-files").replaceChildren();
  controls();
  try {
    const data = await request("/api/v1/models/browse", path ? { path } : {});
    browserLocation = data;
    el("browser-path").value = data.path;
    for (const entry of data.entries) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "file-entry";
      const kind = document.createElement("span");
      kind.className = "file-kind";
      kind.textContent = entry.directory ? "FOLDER" : "GGUF";
      const name = document.createElement("span");
      name.textContent = entry.name;
      button.append(kind, name);
      button.addEventListener("click", () => {
        if (entry.directory) browseModels(entry.path);
        else {
          el("model").value = entry.path;
          modelPrecheck.invalidate();
          modelPrecheck.check();
          el("model-browser").hidden = true;
          el("model").focus();
        }
      });
      item.append(button);
      el("browser-files").append(item);
    }
    el("browser-status").textContent = data.truncated
      ? "This folder is large. Showing a bounded selection; enter a more specific folder path to narrow it."
      : `${data.entries.length} folders / GGUF files${data.entries.length ? "" : " — no models here"}. Select a GGUF to use its path.`;
  } catch (error) {
    el("browser-status").textContent = "Folder could not be opened.";
    el("browser-error").textContent = error.message;
    el("browser-error").hidden = false;
  } finally {
    browserBusy = false;
    controls();
  }
}

el("browse-models").addEventListener("click", () => browseModels(el("model").value.trim() || undefined));
el("close-browser").addEventListener("click", () => {
  el("model-browser").hidden = true;
  el("browse-models").focus();
});
el("browser-go").addEventListener("click", () => browseModels(el("browser-path").value.trim()));
el("browser-home").addEventListener("click", () => browseModels());
el("browser-root").addEventListener("click", () => browseModels(browserLocation?.root));
el("browser-up").addEventListener("click", () => {
  if (browserLocation?.parent) browseModels(browserLocation.parent);
});
el("browser-path").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    browseModels(el("browser-path").value.trim());
  }
});
el("execution-mode").addEventListener("change", () => {
  el("draft-type").value = "none";
  el("draft-model").value = "";
  if (el("execution-mode").value !== "auto") {
    const hybrid = el("execution-mode").value === "hybrid";
    el("context").value = hybrid ? String(status.hybrid_context) : "4096";
    el("placement").value = "auto";
    el("cpu-threads").value = Math.min(8, status.logical_cpus);
    el("batch-threads").value = Math.min(16, status.logical_cpus);
    el("host-op-offload").value = String(!hybrid);
    el("expert-cache").value = hybrid ? String(status.hybrid_cache_mib) : "0";
    el("fit-target").value = "1024";
    el("op-offload-min-batch").value = "32";
    el("load-mode").value = "auto";
    el("draft-model").value = "";
  }
  controls();
});

el("draft-type").addEventListener("change", () => {
  const type = el("draft-type").value;
  if (!["none", "dflash", "mtp"].includes(type)) {
    notice("Select a supported speculative decoding method.");
    return;
  }
  el("draft-tokens").value = type === "mtp" ? "4" : "9";
  el("draft-model").placeholder = type === "mtp" ? "/path/to/native-mtp.gguf" : "/path/to/target-trained-dflash.gguf";
  controls();
});

async function poll() {
  try {
    const next = await request("/api/v1/status");
    const recovered = !status;
    const recoveredRuntime = (status?.error || status?.context_warning) && !next.error && !next.context_warning;
    status = next;
    if (recovered || recoveredRuntime) notice();
    renderStatus();
  } catch (error) {
    status = null;
    el("state").textContent = "Disconnected";
    el("state").dataset.state = "error";
    notice(`Cannot reach the local service: ${error.message}`);
    controls();
  } finally {
    window.setTimeout(poll, 1500);
  }
}

el("endpoint").value = `${window.location.origin}/v1`;
el("connect-example").textContent = `import json
from urllib.request import Request, urlopen

request = Request(
    "${window.location.origin}/v1/chat/completions",
    data=json.dumps({
        "model": "local-model",
        "messages": [{"role": "user", "content": "Hello!"}]
    }).encode(),
    headers={"Content-Type": "application/json"}
)
with urlopen(request) as response:
    print(json.load(response)["choices"][0]["message"]["content"])`;
el("copy-example").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(el("connect-example").textContent);
    el("copy-example").textContent = "Copied";
    window.setTimeout(() => { el("copy-example").textContent = "Copy Python"; }, 1500);
  } catch (error) {
    notice(`Clipboard unavailable. Select and copy the Python example manually. ${error.message}`);
  }
});
el("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(el("endpoint").value);
    el("copy").textContent = "Copied";
    window.setTimeout(() => { el("copy").textContent = "Copy"; }, 1500);
  } catch (error) {
    notice(`Clipboard unavailable. Select and copy the Base URL manually. ${error.message}`);
  }
});

el("load-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  actionPending = true;
  notice();
  controls();
  try {
    const settings = {
      model: el("model").value,
      execution_mode: el("execution-mode").value,
      context: Number(el("context").value),
      placement: el("placement").value,
      cpu_threads: Number(el("cpu-threads").value),
      batch_threads: Number(el("batch-threads").value),
      cpu_poll: Number(el("cpu-poll").value),
      host_op_offload: el("host-op-offload").value === "true",
      op_offload_min_batch: Number(el("op-offload-min-batch").value),
      kv_cache_type: el("kv-cache-type").value,
      load_mode: el("load-mode").value,
      expert_cache_mib: Number(el("expert-cache").value),
      fit_target_mib: Number(el("fit-target").value),
      draft_type: status?.speculation_enabled ? el("draft-type").value : "none",
      draft_model: status?.speculation_enabled && el("draft-type").value !== "none"
        ? el("draft-model").value.trim() || null : null,
      draft_tokens: Number(el("draft-tokens").value),
    };
    const payload = modelLoadPayload(settings);
    status = await request("/api/v1/model/load", payload);
    clearChat();
    el("model-browser").hidden = true;
    el("model-controls").close();
    renderStatus();
  } catch (error) {
    notice(error.message);
    el("controls-action-status").textContent = error.message;
  } finally {
    actionPending = false;
    controls();
  }
});

el("unload").addEventListener("click", async () => {
  controller?.abort();
  actionPending = true;
  controls();
  notice();
  try {
    status = await request("/api/v1/model/unload", {});
    renderStatus();
  } catch (error) {
    notice(error.message);
    el("controls-action-status").textContent = error.message;
  } finally {
    actionPending = false;
    controls();
  }
});

function clearChat() {
  history = [];
  el("messages").replaceChildren();
  const empty = document.createElement("p");
  empty.id = "empty";
  empty.textContent = "Start a new conversation. Open Model Controls to load a model or adjust generation. Your chat is not saved.";
  el("messages").append(empty);
  el("timing").textContent = "";
}

el("clear").addEventListener("click", () => { clearChat(); notice(); });
el("stop").addEventListener("click", () => controller?.abort());

function message(role, text) {
  el("empty")?.remove();
  const item = document.createElement("div");
  item.className = "message";
  item.dataset.role = role;
  const heading = document.createElement("strong");
  heading.textContent = role === "user" ? "You" : "Assistant";
  const content = document.createElement("div");
  content.className = "text";
  content.textContent = text;
  item.append(heading, content);
  el("messages").append(item);
  item.scrollIntoView({ block: "nearest" });
  return content;
}

function assistantMessage() {
  const output = message("assistant", "Waiting for model...");
  const details = document.createElement("details");
  details.className = "reasoning";
  details.hidden = true;
  const summary = document.createElement("summary");
  summary.textContent = "Reasoning";
  const reasoning = document.createElement("div");
  reasoning.className = "text";
  details.append(summary, reasoning);
  output.before(details);
  return { output, details, reasoning, summary };
}

bindChatInput(el("prompt"), el("chat-form"), el("send"));
el("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = el("prompt").value.trim();
  if (!prompt || controller || actionPending || status?.state !== "ready" || status.chat_busy) return;
  const maxTokens = Number(el("max-output").value);
  const reasoningEffort = el("reasoning-effort").value;
  if (!["default", "low", "medium", "high"].includes(reasoningEffort)) {
    notice("Select a supported reasoning effort.");
    return;
  }
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 65536) {
    notice("Max output tokens must be an integer between 1 and 65,536.");
    return;
  }
  notice();
  controller = new AbortController();
  controls();
  el("prompt").value = "";
  message("user", prompt);
  const view = assistantMessage();
  const started = performance.now();
  const previous = [...history, { role: "user", content: prompt }];
  const stream = new ChatStream();
  let phase = "Waiting for model";
  const updateMetrics = () => {
    el("timing").textContent = stream.metrics((performance.now() - started) / 1000, phase);
  };
  const timer = window.setInterval(updateMetrics, 500);
  updateMetrics();
  let reader;
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "local-model", messages: previous, stream: true, max_tokens: maxTokens,
        ...(reasoningEffort === "default" ? {} : { reasoning_effort: reasoningEffort }),
        stream_options: { include_usage: true }, timings_per_token: true,
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error?.message || `Chat failed (${response.status}).`);
    }
    reader = response.body.getReader();
    while (!stream.done) {
      const chunk = await reader.read();
      stream.push(chunk.value, chunk.done);
      phase = stream.content ? "Answering" : stream.reasoning ? "Thinking" : "Waiting for model";
      view.details.hidden = !stream.reasoning;
      view.reasoning.textContent = stream.reasoning;
      view.summary.textContent = stream.content ? "Reasoning" : "Reasoning (in progress)";
      view.output.textContent = stream.content || (stream.reasoning
        ? "Thinking... expand Reasoning to see progress." : "Waiting for model...");
      updateMetrics();
      el("messages").scrollTop = el("messages").scrollHeight;
    }
    phase = stream.finishReason === "length" ? "Token limit reached" : "Complete";
    view.summary.textContent = "Reasoning";
    const warning = stream.warning(maxTokens);
    if (warning) notice(warning);
    if (stream.content.trim()) {
      history = [...previous, { role: "assistant", content: stream.content }];
    } else {
      view.output.textContent = warning;
      el("prompt").value = prompt;
    }
  } catch (error) {
    const interrupted = error.name === "AbortError";
    phase = interrupted ? "Stopped" : "Failed";
    view.summary.textContent = interrupted ? "Reasoning (stopped)" : "Reasoning (interrupted)";
    if (stream.content.trim()) {
      history = [...previous, { role: "assistant", content: stream.content }];
      view.output.textContent = stream.content;
    }
    if (!stream.content.trim()) view.output.textContent = interrupted ? "Generation stopped." : "No final answer.";
    notice(interrupted ? "Generation stopped. Any partial answer is retained." : error.message);
    if (!stream.content.trim()) el("prompt").value = prompt;
  } finally {
    window.clearInterval(timer);
    updateMetrics();
    if (reader) {
      if (!stream.done && !controller.signal.aborted) {
        try { await reader.cancel(); } catch (error) { notice(`Could not close the chat stream: ${error.message}`); }
      }
      reader.releaseLock();
    }
    controller = null;
    controls();
  }
});

window.addEventListener("hashchange", () => navigate());
navigate();
const updateView = new UpdateView(request);
updateView.poll();
new MonitorView(request).poll();
poll();
