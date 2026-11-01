import test from "node:test";
import assert from "node:assert/strict";
import { bindReasoningEffort, draftControls, elapsed, loadingText, modelLoadPayload, ModelPrecheck, precheckText } from "../synapse_app/web/model-controls.mjs";

class Select extends EventTarget {
  constructor(value) { super(); this.value = value; }
  choose(value) { this.value = value; this.dispatchEvent(new Event("change")); }
}

test("reasoning selectors share the initial value and synchronize in both directions", () => {
  const primary = new Select("medium"), quick = new Select("default");
  const errors = [];
  bindReasoningEffort(primary, quick, (error) => errors.push(error));
  assert.equal(quick.value, "medium");
  quick.choose("high");
  assert.equal(primary.value, "high");
  primary.choose("low");
  assert.equal(quick.value, "low");
  quick.choose("default");
  assert.equal(primary.value, "default");
  assert.deepEqual(errors, []);
});

test("invalid reasoning effort is reported and cannot change the request selector", () => {
  const primary = new Select("default"), quick = new Select("default"), errors = [];
  bindReasoningEffort(primary, quick, (error) => errors.push(error));
  quick.choose("unlimited");
  assert.equal(primary.value, "default");
  assert.equal(errors.length, 1);
});

test("loading feedback distinguishes verification, elapsed time and quiet output without a fake ETA", () => {
  const text = loadingText({ phase: "Verifying experimental hybrid execution", elapsed_seconds: 125,
    last_output_age_seconds: 60, waiting_for: ["full routed-layer coverage"] });
  assert.match(text, /2m 5s elapsed/);
  assert.match(text, /Waiting for full routed-layer coverage/);
  assert.match(text, /No new backend output for 1m 0s/);
  assert.doesNotMatch(text, /remaining|%/);
  assert.equal(loadingText(null), "");
  assert.equal(elapsed(59.9), "59s");
});

const candidate = {
  architecture: "future-moe", hybrid_candidate: true, layout: { expert_count: 7, block_count: 3 },
  automatic: { execution_profile: "standard", reason: "Native fitting is preferred for this small model." },
  required_hybrid: { initialization: "causal", transfer: "staged", context: 16384, cache_mib: 8192 },
  scope: "Read-only metadata; no backend started.",
  limitations: ["Kernel support and memory fit are unverified.", "Speed and quality are unverified."],
};

test("pre-check separates candidate eligibility from Automatic's actual selection and unverified capabilities", () => {
  const text = precheckText(candidate);
  assert.match(text, /Candidate - runtime unverified/);
  assert.match(text, /Automatic defaults: Standard backend/);
  assert.match(text, /Require hybrid: causal plan, staged transfers/);
  assert.match(text, /Kernel support and memory fit are unverified/);
  assert.match(text, /Speed and quality are unverified/);
  const unavailable = precheckText({ ...candidate, hybrid_candidate: false,
    required_hybrid: { reason: "No hybrid installation profile found." } });
  assert.match(unavailable, /Not eligible with this installation/);
  assert.match(unavailable, /No hybrid installation profile/);
});

function precheck(request) {
  const input = new Select("/models/example.gguf"), button = new EventTarget(), output = { textContent: "" };
  const view = new ModelPrecheck({ input, button, output, request, changed: () => {} });
  return { input, button, output, view };
}

test("pre-check sends only the selected path, displays results and surfaces errors", async () => {
  const calls = [];
  const { view, output } = precheck(async (...args) => { calls.push(args); return candidate; });
  await view.check();
  assert.deepEqual(calls, [["/api/v1/model/precheck", { model: "/models/example.gguf" }]]);
  assert.match(output.textContent, /Hybrid eligibility: Candidate/);
  assert.equal(view.busy, false);
  const failing = precheck(async () => { throw new Error("GGUF metadata is truncated."); });
  await failing.view.check();
  assert.equal(failing.output.textContent, "Pre-check failed: GGUF metadata is truncated.");
  assert.equal(failing.view.busy, false);
});

test("editing the path invalidates results and prevents stale requests from replacing a newer check", async () => {
  const pending = [];
  const { view, input, output } = precheck(() => new Promise((resolve, reject) => pending.push({ resolve, reject })));
  const first = view.check();
  input.value = "/models/second.gguf";
  input.dispatchEvent(new Event("input"));
  assert.match(output.textContent, /Model path changed/);
  const second = view.check();
  pending[0].resolve(candidate);
  await first;
  assert.equal(view.busy, true);
  assert.match(output.textContent, /Reading GGUF metadata/);
  pending[1].resolve({ ...candidate, architecture: "second" });
  await second;
  assert.match(output.textContent, /Architecture: second/);
  input.value = "/models/third.gguf";
  const third = view.check();
  input.dispatchEvent(new Event("input"));
  pending[2].reject(new Error("obsolete failure"));
  await third;
  assert.doesNotMatch(output.textContent, /obsolete failure/);
});

test("empty paths do not send inspection requests", async () => {
  const { view, input, output } = precheck(() => assert.fail("Unexpected request"));
  input.value = " ";
  await view.check();
  assert.match(output.textContent, /Enter a local GGUF path/);
  assert.equal(view.busy, false);
});

test("draft controls preserve default-off behavior and restrict Hybrid to native MTP", () => {
  const status = { speculation_enabled: true, backend_kind: "llama.cpp" };
  assert.equal(draftControls({ ...status, speculation_enabled: false }, "standard", "mtp").disabled, true);
  assert.equal(draftControls(status, "standard", "none").inputDisabled, true);
  assert.equal(draftControls(status, "standard", "dflash").inputDisabled, false);
  assert.equal(draftControls(status, "hybrid", "dflash").inputDisabled, true);
  assert.equal(draftControls(status, "hybrid", "mtp").inputDisabled, false);
  assert.equal(draftControls(status, "auto", "mtp").inputDisabled, false);
  assert.equal(draftControls(status, "hybrid", "mtp").maximum, 31);
  assert.equal(draftControls(status, "standard", "mtp").maximum, 32);
  const experimental = draftControls({ ...status, backend_kind: "ik_llama.cpp" }, "standard", "mtp");
  assert.equal(experimental.mtpDisabled, true);
  assert.equal(experimental.inputDisabled, true);
});

test("Automatic and Hybrid payloads retain the explicit MTP draft instead of silently dropping it", () => {
  const settings = { model: "/models/target.gguf", execution_mode: "hybrid", context: 16384,
    cpu_threads: 8, batch_threads: 16, cpu_poll: 50, expert_cache_mib: 6144, kv_cache_type: "f16",
    draft_type: "mtp", draft_model: "/models/mtp.gguf", draft_tokens: 4 };
  assert.deepEqual(modelLoadPayload(settings), {
    model: settings.model, execution_mode: "hybrid", kv_cache_type: "f16",
    draft_type: "mtp", draft_model: settings.draft_model, draft_tokens: 4,
    cpu_threads: 8, batch_threads: 16, cpu_poll: 50, expert_cache_mib: 6144,
  });
  assert.deepEqual(modelLoadPayload({ ...settings, execution_mode: "auto" }), {
    model: settings.model, execution_mode: "auto", kv_cache_type: "f16",
    draft_type: "mtp", draft_model: settings.draft_model, draft_tokens: 4,
  });
  const standard = { ...settings, execution_mode: "standard", draft_type: "dflash" };
  assert.equal(modelLoadPayload(standard), standard);
});
