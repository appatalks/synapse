export function elapsed(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  return total < 60 ? `${total}s` : `${Math.floor(total / 60)}m ${total % 60}s`;
}

export function loadingText(loading) {
  if (!loading) return "";
  const pending = loading.waiting_for?.length ? ` Waiting for ${loading.waiting_for.join(", ")}.` : "";
  const quiet = loading.last_output_age_seconds >= 15
    ? ` No new backend output for ${elapsed(loading.last_output_age_seconds)}; readiness checks continue.` : "";
  return `${loading.phase} · ${elapsed(loading.elapsed_seconds)} elapsed.${pending}${quiet}`;
}

export function draftControls(status, mode, type) {
  const enabled = !!status?.speculation_enabled;
  const mtp = status?.backend_kind === "llama.cpp";
  return {
    disabled: !enabled,
    dflashDisabled: mode !== "standard",
    mtpDisabled: !mtp,
    inputDisabled: !enabled || type === "none" ||
      (type === "dflash" && mode !== "standard") || (type === "mtp" && !mtp),
    maximum: type === "mtp" && mode !== "standard" ? 31 : 32,
  };
}

export function modelLoadPayload(settings) {
  if (settings.execution_mode === "standard") return settings;
  const payload = {
    model: settings.model, execution_mode: settings.execution_mode, kv_cache_type: settings.kv_cache_type,
    draft_type: settings.draft_type, draft_model: settings.draft_model, draft_tokens: settings.draft_tokens,
  };
  if (settings.execution_mode === "hybrid") Object.assign(payload, {
    cpu_threads: settings.cpu_threads, batch_threads: settings.batch_threads, cpu_poll: settings.cpu_poll,
    expert_cache_mib: settings.expert_cache_mib,
  });
  return payload;
}

export function bindReasoningEffort(primary, quick, invalid) {
  const values = new Set(["default", "low", "medium", "high"]);
  quick.value = primary.value;
  for (const [source, target] of [[primary, quick], [quick, primary]]) {
    source.addEventListener("change", () => {
      if (!values.has(source.value)) {
        invalid("Select a supported reasoning effort.");
        return;
      }
      target.value = source.value;
    });
  }
}

export function precheckText(data) {
  const automatic = data.automatic, hybrid = data.required_hybrid;
  const lines = [
    `Hybrid eligibility: ${data.hybrid_candidate ? "Candidate - runtime unverified" : "Not eligible with this installation"}.`,
    `Architecture: ${data.architecture}${data.layout ? `; ${data.layout.expert_count} experts, ${data.layout.block_count} blocks` : ""}.`,
    `Automatic defaults: ${automatic.execution_profile === "standard" ? "Standard backend" : "Hybrid candidate"}. ${automatic.reason}`,
    `Require hybrid: ${data.hybrid_candidate ? `${hybrid.initialization} plan, ${hybrid.transfer} transfers, ${hybrid.context.toLocaleString()} context, ${hybrid.cache_mib.toLocaleString()} MiB cache.` : hybrid.reason}`,
    data.scope,
    ...data.limitations,
  ];
  return lines.join("\n\n");
}

export class ModelPrecheck {
  constructor({ input, button, output, request, changed }) {
    Object.assign(this, { input, button, output, request, changed });
    this.serial = 0;
    this.busy = false;
    button.addEventListener("click", () => this.check());
    input.addEventListener("input", () => this.invalidate());
    input.addEventListener("change", () => this.check());
  }

  invalidate() {
    this.serial += 1;
    this.busy = false;
    this.output.textContent = "Model path changed. Check hybrid eligibility for the selected file.";
    this.changed();
  }

  async check() {
    const model = this.input.value.trim(), serial = ++this.serial;
    this.busy = false;
    if (!model) {
      this.output.textContent = "Enter a local GGUF path to check hybrid eligibility.";
      this.changed();
      return;
    }
    this.busy = true;
    this.output.textContent = "Reading GGUF metadata and installation policy; no model is being loaded...";
    this.changed();
    try {
      const data = await this.request("/api/v1/model/precheck", { model });
      if (serial === this.serial && model === this.input.value.trim()) this.output.textContent = precheckText(data);
    } catch (error) {
      if (serial === this.serial && model === this.input.value.trim()) {
        this.output.textContent = `Pre-check failed: ${error.message}`;
      }
    } finally {
      if (serial === this.serial) {
        this.busy = false;
        this.changed();
      }
    }
  }
}
