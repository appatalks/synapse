# Application notes

[Back to SYNAPSE](../README.md).

## Runtime requirements and local data

No account, hosted inference or model uploads. Chats stay in the page.
The app needs only Python and a browser; models and an appropriate
[llama.cpp backend](https://github.com/ggml-org/llama.cpp) are separate.
The reference backend revision is b11669.
[Platform and sensor limits](monitoring.md).

## Hybrid eligibility pre-check

In **Chat → Model Controls**, select a local GGUF in the browser or enter its
path and choose **Check hybrid eligibility**. Selecting a file or finishing a
path edit also runs the check. It reads bounded GGUF metadata and the installed
profile policy without mapping weights, starting a backend, allocating a GPU
cache, or unloading the active model. Select the first shard of a split GGUF.

The result separates two questions:

- **Hybrid candidate:** the file has a complete routed-MoE descriptor accepted
  by the installed plan policy. Generic version-2 profiles use dimensions, not
  a model-name allowlist; an unfamiliar architecture can still be a candidate.
  A candidate is **not** a tested or quality-qualified configuration.
- **Automatic defaults:** what Automatic would select now. It may prefer native
  fitting for a small single-file MoE even when Require hybrid is a candidate.
  The cache-budget heuristic does not establish actual VRAM fit or speed.

When there is no candidate, the result explains whether the installation lacks
a hybrid profile, the descriptor is missing/incomplete, or the model does not
match a bound training plan and generic causal dispatch is disabled. Missing
expert metadata does not prove a model is dense.

Kernel/operation support, tensor layouts, precision requirements, all split
shards, available RAM/VRAM, runtime activation and layer coverage remain
**unverified until loading**. An unsupported operation can still reject a
candidate; the Mistral study is one example. If a selected hybrid load fails,
the error is reported rather than silently retrying as standard execution.
No pre-check establishes a speedup or equivalent answer quality.

The read-only API is `POST /api/v1/model/precheck` with
`{"model": "/absolute/path/to/model.gguf"}`. It returns `hybrid_candidate`,
the detected `architecture`/`layout`, separate `automatic` and `required_hybrid`
decisions with `reason_code`, default context/cache/plan/transfer information,
and explicit `scope`/`limitations`. It evaluates profile defaults, not manual
run settings. Invalid or unreadable GGUF metadata returns an error, not a
standard-mode success result. Changing the path invalidates the displayed
result; checks run again at load time.

## Auto Tune

**Auto Tune** in Model Controls measures near-capacity input on your system and
applies a passing configuration, then offers copyable VS Code/OpenAI-client limits.
Balanced daily use tests up to 32K; Maximum Context can test up to the model limit
or 256K and take an hour or more. Experimental hybrid trials require a separate
opt-in. Tuning temporarily owns the model; cancellation restores the previous
configuration when possible. It tests input capacity, short anchor-retrieval
replies and fixed-length generation speed for every engine the installation
supports for the model. Balanced picks the faster generator and Maximum Context
the faster long-prompt run; Automatic then reuses that choice on this machine
([automatic engine choice](installing.md#automatic-engine-choice)). It is not a
general answer-quality check or an exhaustive speed search.

[Tuning scope, saved results and client settings](installing.md#auto-tune) ·
[Optional Q8 attention cache](installing.md#optional-q8-attention-cache).

## Experimental speculative decoding

Start SYNAPSE with `--experimental-speculation` to enable the explicit
**Speculative decoding** selector under **Model Controls -> Advanced performance**.
It stays Off unless you select a method and a matching local draft GGUF.

- **DFlash:** Standard execution only, with a compatible target-trained DFlash
  draft. Existing API requests supplying `draft_model` without `draft_type`
  retain DFlash behavior.
- **Native MTP:** the official llama.cpp adapter supports Standard, Automatic,
  and experimental Hybrid execution with a matching native nextn draft. Strata's
  `qwen4exp-mtp` files and runtime directories are not native MTP drafts.
  Architecture, embedding dimensions, declared vocabulary sizes, and nextn
  metadata are checked before launch; the backend must validate the remaining
  tensor/tokenizer compatibility.

The API uses `draft_type: "none" | "dflash" | "mtp"`, `draft_model`, and
`draft_tokens`. MTP defaults to at most four proposals per verification.
Hybrid MTP allows at most 31 proposals plus one target token, explicitly opts
into the native hybrid-MTP path, and keeps the expert bank in the target context
rather than the draft context. Every target slot must report active speculation
before Ready; a backend that falls back to target-only execution fails verification.

Drafts require additional VRAM. A target-only expert bank that fits may fail once
a draft is added; reduce an unbound causal bank explicitly in Require hybrid
mode rather than expecting an automatic fallback. Installed profiles and default
execution settings are not rewritten. Auto Tune preserves the selected method
when restoring a previous model, but its capacity trials do not benchmark drafting.

MTP drafts may come from the stock model even when the target is modified.
Acceptance, fit and speed must be measured against that exact target. Activation
checks and accepted-token timings do not establish native-reference equivalence,
general answer quality, or refusal behavior. Hybrid MTP remains experimental.

Successful activation is not a speed recommendation. The local SC117 IQ3_S
screen on the RTX 5070 Ti 16GB, at 16K context with four draft proposals, was
slower with MTP in both Standard and Hybrid execution. Keep drafting Off unless
benchmarks of your exact target, settings and hardware demonstrate a benefit.

## Updates without reinstalling


Click the **version in the sidebar**. It lights up when a newer compatible
public release is found. Managed installs can download a verified update and
restart after your confirmation. Models, backends and local profiles are not
replaced. Source/pip installs get release information, not overwritten files.

Update checks contact GitHub for release metadata only, never prompts or
hardware data. Use `--no-update-check` to disable automatic checks.
Checksums verify the downloaded artifact's integrity; they are not a separate
publisher signature. [Installation and update details](installing.md).

## Benchmark interpretation and experimental compatibility

The [README performance table](../README.md#why-use-it) contains matched
three-run, 512-token technical-screen medians at 16K context, not universal
speed or quality guarantees. The monitoring screenshot illustrates the UI,
not a separate benchmark result.

Later GPT tuning reached a separate 50.06 tok/s technical median, but that is
**not** the matched comparison in the README or a sustained cross-workload
60 tok/s result.

Not every MoE is hybrid-compatible: the
[Mistral Small 4 study](benchmarks/mistral/mistral4-compatibility.md) records its FP32
operation requirement and unpromoted prototype. Its
[isolated native integration and research timings](benchmarks/mistral/mistral4-native-experimental.md)
are experimental only: strict native equivalence still fails, so it is not
included in the README's headline performance table.

[Detailed compatibility status and evidence](compatibility.md).

## Engine lineage

The engine builds on llama.cpp's upstream expert-cache machinery, including
[ggml-org/llama.cpp#29887](https://github.com/ggml-org/llama.cpp/pull/29887),
already included in our pinned backend. That cache uploads misses for GPU
execution; SYNAPSE's experimental hybrid dispatcher extends the tensor/slot
machinery with CPU cold-expert execution and guarded causal admissions.
The PR's published speedups are not an additional multiplier on SYNAPSE's results.
[Local upstream-cache investigation and the pinned-host-loading result](benchmarks/gpt-oss/upstream-pr29887.md).

The experiments, native patches, benchmark methodology and full history live in
[unified-memory-scheduler](https://github.com/appatalks/unified-memory-scheduler).

## Local-only API

The API has no app authentication. Do not expose it through a tunnel or reverse
proxy. No universal speedup, fit guarantee or quality qualification is implied.

[Installation](installing.md) · [Monitoring](monitoring.md) ·
[Development](releasing.md) · [MIT license](../LICENSE).
