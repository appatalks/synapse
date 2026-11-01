# Project Guidelines

## Sister projects: research vs. production

- **This repository (appatalks/Synapse) is the production application.** It ships only behavior that has been
  measured and qualified. New engine techniques arrive here *after* they are proven in research.
- **[appatalks/unified-memory-scheduler](https://github.com/appatalks/unified-memory-scheduler) is the research
  area** (VRAMBridge): backend patches, experiments, benchmarks, raw results and negative findings.
- Flow of knowledge: investigate and measure in unified-memory-scheduler → record the evidence and the canonical
  patch there → apply the *learned, validated* subset here (defaults, guards, opt-ins, docs, tests).
- Do not develop experimental engine code in Synapse. Do not leave validated production behavior only in research.
- Before changing behavior that depends on the backend, read the current research state (patch, `benchmarks/README.md`)
  rather than relying on memory of earlier sessions.

## Cross-repo release order

[scripts/setup_acceleration.py](../scripts/setup_acceleration.py) pins a research commit (`RESEARCH`) and the
patch checksum (`PATCH_SHA`). When the backend patch changes:

1. Commit and push unified-memory-scheduler first.
2. Verify the pinned raw URL serves the expected SHA-256.
3. Bump `RESEARCH`/`PATCH_SHA` here and commit Synapse.

Ask the user before committing or pushing either repository.

## Product principles

- **Model-agnostic.** A speed change must be validated on several MoE families (at minimum Qwen3-Coder-30B-A3B,
  a many-expert Qwen3.8 model, and gpt-oss-120b) and must not regress the others. Never key behavior on model names;
  derive it from GGUF metadata and tensors.
- **Accurate only.** No lossy shortcuts (skipped layers, pruned experts, changed weights) for speed.
- **No silent fallback.** An experimental path must be confirmed by backend log markers
  (`required_hybrid_markers`); if they are missing, loading fails with an explanation instead of quietly running
  something else.
- **Engine internals are invisible to users.** Standard, hybrid (GPU + CPU expert sharing), classic vs. hosted
  (in-graph) methods, admission policies and similar are internal. Users keep **Engine: Automatic**; the app
  detects what the installation supports (`engine_capabilities`), starts from safe defaults, and Auto Tune measures
  every supported engine and remembers the winner per model + machine + backend build (`EngineChoices`). New engine
  techniques plug into that detect → measure → remember flow, not into new user-facing toggles. Manual choices
  exist only for troubleshooting.
- **Chosen by measurement until it wins broadly.** Techniques that help some models and hurt others are never the
  untested default; Automatic uses them only where Auto Tune measured them faster, with results in the docs.
- Do not change the user's installed backend, profile or saved state (`~/.config/synapse`,
  `~/.local/share/vrambridge`) without consent. Use a temporary `XDG_CONFIG_HOME` for experiments.
- Auto Tune ([synapse_app/tuning.py](../synapse_app/tuning.py)) must work on any system: CPU-only, small GPUs,
  Windows/macOS (no hybrid), and Linux/NVIDIA with or without the hybrid profile.

## Build and test

- `python tests/test_local_app.py synapse_app/app.py` (backend lifecycle; uses
  [tests/fake_llama_server.py](../tests/fake_llama_server.py), which must emit any new backend markers).
- `python -m unittest discover -s tests -p test_tuning.py`
- `node --test tests/*.mjs`
- `python scripts/build_synapse.py --output dist && python tests/test_release.py dist/SYNAPSE.pyz`
- See [docs/releasing.md](../docs/releasing.md) and [.github/workflows/ci.yml](workflows/ci.yml).

## Local hardware and models

- Linux dev machine: Ryzen 9 9950X (16C/32T), ~126 GB RAM, RTX 5070 Ti 16 GB (sm_120).
- Model roots: `/home/mj420/Desktop/TOR/llm` and `/media/mj420/ExpansionXFS1/ai-models` (archive library at
  `.../ai-models/synapse-library`). Keep weights out of the repository.
- Stop only processes you started; never `pkill` every `llama-server`.
