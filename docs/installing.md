# Installation and updates

## One-line Linux/macOS install

```bash
curl -fsSL https://raw.githubusercontent.com/appatalks/synapse/main/install.sh | bash -s -- --launch
```

Requirements: Python 3.10+, bash, curl, a browser and your own GGUF model.
The script bootstraps from this repository over HTTPS, prefers a public
checksummed zipapp, and clearly labels a source preview if no release exists.
Review the script first if you do not want to pipe a network script into bash.

- Application: `~/.local/share/synapse/SYNAPSE.pyz`
- Launcher: `~/.local/bin/synapse` (use this full path if that folder is not on `PATH`)
- No sudo, driver installation, model downloads or changes to your system Python.
- Supply an existing backend with `--llama-server /path/to/llama-server`.
- `--prefix /private/folder` puts both app and launcher under that folder.
- For an offline/local build, run `python install.py --local-package dist/SYNAPSE.pyz --no-accelerated`.

The app opens `http://127.0.0.1:8080`. Ctrl+C stops it and its owned backend.
`--port 0` chooses a free port; `--no-browser` suppresses browser opening.

## Context size and editor clients

In **Chat -> Model Controls -> Run settings**, standard execution accepts
512 through 262,144 context tokens (256K). Unload and reload the model to change
its allocation. This is an application limit, not a fit, speed or model-support
guarantee. Try larger contexts progressively and check the reported loaded
context, memory use and a real long-prompt response, not just successful loading.
Automatic fitting may reduce GPU weight placement as context memory grows.

Experimental hybrid execution still uses the context budget in its installation
profile, capped by model metadata. Profiles may explicitly configure up to
262,144 tokens, but existing defaults and exact-context checks are unchanged.
Hybrid fitting is disabled: larger contexts compete with the fixed GPU expert
bank and may fail to allocate. Preserve the original profile before experimenting;
successful allocation or generation does not establish answer quality.

For VS Code or another OpenAI-compatible client, advertise limits that fit the
backend's **actual loaded context**, allowing room for output and token-count
differences. Editor settings do not increase the server allocation. For example,
a 32,768-token server can start with 24,576 input and 4,096 output tokens.
The browser chat's output limit is separate from an editor client's output limit.
JSON request bodies are limited to 8 MiB so long prompts can pass through the API;
the backend still enforces its token limit.

### Optional Q8 attention cache

Under **Run settings -> Advanced performance**, the official llama.cpp backend
offers an explicit **Q8 K/V + Flash Attention** preset. FP16 backend defaults
remain unchanged unless selected. The preset passes `--cache-type-k q8_0`,
`--cache-type-v q8_0` and `--flash-attn on`; it does not reduce context length
or change model weights. Auto Tune retains the selected preset through trials,
application and restoration.

Q8 can roughly halve attention K/V storage compared with FP16, but does not
halve total model memory or quantize every recurrent-state buffer. It changes
cache precision and can affect answer quality. Extra memory headroom may improve
placement or enable a larger hybrid expert bank; speedups are not guaranteed.
SYNAPSE requires backend confirmation of both Q8 attention-cache types and enabled
Flash Attention before marking this preset ready. A reported precision fallback
fails explicitly. Other backend adapters retain their existing defaults.
The Q8 preset also enables the pinned backend's library-info diagnostics
(`--log-verbosity 4`) so these confirmations are visible. Debug and verbose-prompt
logging are not enabled, and backend authentication diagnostic lines are redacted.

### Optional native MTP drafting

Launch with `--experimental-speculation`, then choose **Native MTP** under
**Model Controls -> Advanced performance**, select a compatible native MTP
GGUF, and start with four draft tokens. The official llama.cpp backend must
support `draft-mtp` and report an active speculative decoder through `/slots`.
The existing DFlash method remains Standard-only; drafting stays Off by default.

Hybrid MTP also requires the matching patched backend's native MTP and
multi-position hybrid dispatch hooks. It verifies up to 31 proposals plus one
target token and needs extra draft VRAM. Reduce the causal expert-bank budget
explicitly if the combined target/draft does not fit. No smaller context, changed
weights, or target-only fallback is silently substituted.

[Draft compatibility, API fields and qualification limits](application_notes.md#experimental-speculative-decoding).

## Auto Tune

Open **Model Controls -> Auto Tune for this system**, choose a goal and explicitly
confirm that SYNAPSE may unload/reload the selected GGUF and run local benchmarks.
Balanced tests progressively through 8K, 16K and 32K, capped by model metadata.
Maximum Context continues through 64K, 128K and 256K where previous trials pass.
A near-capacity synthetic prompt must produce a complete streamed response,
backend-reported input usage and the correct start/middle/end anchors.
Successful loading alone never qualifies a context as verified.

Tuning uses automatic backend fitting (or explicitly selected CPU placement),
up to 8 generation / 16 prompt threads within CPU topology/affinity, and a
1024 MiB VRAM reserve. Thread counts are derived defaults, not individually
benchmarked optima. This is a context-capacity tuner, not an exhaustive search
of threads, quantization, loading modes or cache policies. Long prompts can take
minutes each; Maximum Context can take an hour or more. RAM availability is
checked, but OS availability includes reclaimable cache and is not a fit guarantee.
Allocation failures are retained in the results rather than hidden.

Standard execution is the default. The optional **Include experimental hybrid
trials** checkbox tests the largest passing standard context using the installed
compatible profile. Unbound causal layouts may try progressively smaller expert
banks; bound training plans retain their budget. Profiles and training plans are
not rewritten. A passing hybrid trial is selected only if its prompt-plus-short-
answer time is at least 5% lower; this does not establish broad answer quality.

Other chat/load/unload/update operations are blocked during a run. **Cancel &
restore** cancels a probe and attempts to reload the previous configuration.
Application shutdown cancels tuning without starting another model. Failed
restoration is reported explicitly, including when RAM headroom prevents it.

After completion, recommended model settings and the browser's output allowance
are applied, and the UI shows
copyable VS Code `maxInputTokens` / `maxOutputTokens`, the backend context,
model ID and safety margin. Other apps are never edited automatically.
Recommendations are marked stale and cannot be copied as current limits when
the active model/context no longer matches the measured settings. The output
allowance is a conservative allocation, not a sustained long-generation test.

The latest report is saved privately in `tuning.json` under
`$XDG_CONFIG_HOME/synapse` (normally `~/.config/synapse`) on Linux,
`~/Library/Application Support/SYNAPSE` on macOS, or `%LOCALAPPDATA%/SYNAPSE`
on Windows. It contains the local model path, hardware summary, settings and
measurements, not prompt text or chat history. Page refresh/app restart retains
the report but does not automatically load its model.

## Loading feedback

The loading banner remains visible outside Model Controls and shows the actual
phase, elapsed time, latest backend output and pending experimental checks.
Its progress bar is indeterminate unless actual prompt-processing progress is
reported; there is no invented percentage or completion-time estimate.
**Cancel loading** unloads the in-progress backend without closing the app.

Backend API readiness and experimental activation are separate checks. If the API
is ready but required hybrid activation markers remain missing for 60 seconds,
loading fails with the unconfirmed checks instead of appearing to load indefinitely.
The safety checks are not bypassed. Normal weight loading retains its existing
overall timeout.

## Optional accelerated engine

On Linux the installer offers acceleration through the terminal. Use
`--accelerated` for an explicit non-interactive request or `--no-accelerated`
to skip the offer. It never silently installs system dependencies.

An existing legacy VRAMBridge profile is validated and reused. Otherwise the
setup requires Git, CMake, a C++ compiler, NVIDIA drivers, `nvidia-smi` and `nvcc`.
It downloads the exact llama.cpp source revision and checksummed research patch,
builds matching CPU/CUDA libraries, and writes a private profile. Building can
take several minutes and substantial disk space. Partial failed builds are
retained for diagnosis, not silently erased.

The conservative profile uses causal initialization and bounded staging.
On systems with at least 16,000 MiB reported VRAM and 96 GiB RAM, it enables the
research direct-DMA budgets and a training plan bound to the exact GPT layout;
other routed layouts use their own causal initialization, never that plan.
Actual tensors, capacities, registration/headroom and activation are validated
again by the backend. Desktop load and model/context size can still cause OOM.

Profiles can opt into hosted MoE execution with `"hosted_moe": true`. CPU
experts then run inside the CUDA graph through a pinned mailbox while the GPU
computes resident experts, with frequency-gated admissions (free slot after one
use, replacement after six). It requires direct transfers and the CUDA plugin
built from the patched checkout; it is rejected with MTP drafting until that
pairing is validated. Loading fails unless the backend confirms hosted
execution and the LFU policy. In local screens it was about 30% faster on
Qwen3-Coder-30B-A3B, neutral on a 512-expert Qwen3.8 model, and about 6% slower
on gpt-oss-120b, so it stays off by default.

This is **experimental Linux/NVIDIA code**, not a production-qualified model
executor. CPU/native numerical and answer checks remain unresolved. No model
weights are downloaded, converted or pruned by setup. Use standard execution
with `--execution-mode standard` to bypass the experimental profile.
The measured 16GB GPU profile pins substantial RAM; do not assume a small
machine can reproduce the README's large-model results.

## Windows and source installs

For a standalone app, run the release `SYNAPSE.pyz` with Python:

```powershell
py SYNAPSE.pyz --llama-server C:\path\to\llama-server.exe
```

For a managed install with in-app updates before/after the first release:

```powershell
git clone https://github.com/appatalks/synapse.git
cd synapse
py install.py --no-accelerated --llama-server C:\path\to\llama-server.exe
```

Run the printed `synapse.cmd` launcher. Linux/macOS can also clone the repository
and run `python3 install.py`. Ordinary development still uses
`python -m synapse_app`; that source tree is never modified by in-app updates.

## Release details and self-update

The sidebar version opens a release-details dialog. A mint indicator means an
eligible public release is newer (or a matching public release replaces a
preview). Stable and `a`/`b`/`rc` semantic versions are ordered explicitly;
drafts, invalid tags and releases without both required assets are ignored.

Checks run at most hourly through the fixed `appatalks/synapse` GitHub API;
manual checks can be requested in the dialog. They send no model, prompt or
hardware information. Network errors are shown in that dialog, not reported as
“up to date.” Disable automatic checks with `--no-update-check`.

Only the private, user-owned managed zipapp is eligible for in-place updates.
SYNAPSE verifies a bounded download against the release's `SHA256SUMS`, checks
ZIP structure and the embedded version, and atomically replaces the app.
The prior app is retained as `SYNAPSE.pyz.previous`. No archive paths are
extracted onto the filesystem.

The download/checksum and metadata come from the project's GitHub account over
HTTPS. This verifies integrity, not an independent signing identity; you trust
the repository/release maintainers just as with the installer.

Installing requires confirmation that the model will be unloaded and the app
restarted. A busy request/model transition blocks installation; if a request
starts during download, the update fails explicitly without interrupting it.
The browser can reconnect on the same port. Chats remain in the open page,
but the model must be loaded again. Models, native backends and profiles are
not self-updated. Source, wheel and pipx installations show release details
and must be updated through their original installation method.
