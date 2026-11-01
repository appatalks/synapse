"""Opt-in, local context-capacity trials with conservative client budgets."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import http.client
import json
import logging
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
from typing import Callable


LOG = logging.getLogger("synapse.tuning")
LOAD_FIELDS = ("model", "execution_mode", "context", "placement", "cpu_threads", "batch_threads",
               "load_mode", "expert_cache_mib", "fit_target_mib", "host_op_offload", "cpu_poll",
               "op_offload_min_batch", "kv_cache_type", "draft_model", "draft_type", "draft_tokens",
               "hybrid_variant")
CHOICE_FIELDS = ("execution_mode", "hybrid_variant", "context", "placement", "cpu_threads", "batch_threads",
                 "expert_cache_mib", "kv_cache_type")
MAX_CHOICES = 256
ANCHORS = {"start": "cedar4821", "middle": "violet7319", "end": "quartz9067"}
DECODE_PROMPT = "Write a detailed technical explanation of how a hash table handles collisions.\n\n"
DECODE_CHUNK_TOKENS = 128
DECODE_MAX_CHUNKS = 6


class TrialError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


def _valid_settings(settings: object) -> bool:
    return (isinstance(settings, dict) and type(settings.get("context")) is int and
            512 <= settings["context"] <= 262144 and settings.get("execution_mode") in ("standard", "hybrid") and
            settings.get("draft_type", "none") in ("none", "dflash", "mtp") and
            settings.get("hybrid_variant", "classic") in ("auto", "classic", "hosted") and
            settings.get("kv_cache_type", "f16") in ("f16", "q8_0"))


def client_settings(context: int) -> dict:
    reserve = max(128, min(8192, context // 8))
    margin = max(128, min(8192, context // 8))
    return {"context": context, "maxInputTokens": context - reserve - margin,
            "maxOutputTokens": reserve, "safetyMarginTokens": margin, "model": "local-model"}


def _valid_choice(entry: object) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("settings"), dict):
        return False
    settings = entry["settings"]
    return (set(settings) == set(CHOICE_FIELDS) and _valid_settings(settings) and
            settings["hybrid_variant"] in ("classic", "hosted") and settings["placement"] in ("auto", "cpu") and
            all(type(settings[key]) is int and settings[key] >= 0
                for key in ("cpu_threads", "batch_threads", "expert_cache_mib")) and
            type(entry.get("decode_tokens_per_second")) in (int, float) and
            type(entry.get("standard_decode_tokens_per_second")) in (int, float) and
            type(entry.get("measured_at")) in (int, float) and entry.get("goal") in ("balanced", "maximum"))


class EngineChoices:
    """Per model and machine, the execution choice Auto Tune measured as best; Automatic loads reuse it."""

    def __init__(self, store: Path | None):
        self.store = store
        self.lock = threading.Lock()
        self.entries: dict[str, dict] = {}
        self.error: str | None = None
        if store is not None and store.exists():
            try:
                if store.stat().st_size > 1024 * 1024:
                    raise ValueError("Saved engine choices exceed 1 MiB.")
                saved = json.loads(store.read_text(encoding="utf-8"))
                if (not isinstance(saved, dict) or saved.get("schema_version") != 1 or
                        not isinstance(saved.get("entries"), dict) or len(saved["entries"]) > MAX_CHOICES or
                        not all(isinstance(key, str) and _valid_choice(entry)
                                for key, entry in saved["entries"].items())):
                    raise ValueError("Saved engine choices have an unsupported format.")
                self.entries = saved["entries"]
            except (OSError, ValueError) as exc:
                LOG.error("Saved engine choices unavailable: %s", exc)
                self.error = f"Saved engine choices unavailable: {exc}"

    @staticmethod
    def key(model: Path, kv_cache_type: str, machine: dict) -> str:
        info = model.stat()
        identity = {"model": str(model.resolve()), "size": info.st_size, "mtime_ns": info.st_mtime_ns,
                    "kv_cache_type": kv_cache_type, "machine": machine}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def lookup(self, model: Path, kv_cache_type: str, machine: dict | None) -> dict | None:
        if machine is None:
            return None
        try:
            key = self.key(model, kv_cache_type, machine)
        except OSError:
            return None
        with self.lock:
            entry = self.entries.get(key)
            return deepcopy(entry) if entry is not None else None

    def remember(self, model: Path, machine: dict, entry: dict) -> None:
        if not _valid_choice(entry):
            raise ValueError("Refusing to save an invalid engine choice.")
        key = self.key(model, entry["settings"]["kv_cache_type"], machine)
        with self.lock:
            entries = {**self.entries, key: deepcopy(entry)}
            while len(entries) > MAX_CHOICES:
                del entries[min(entries, key=lambda name: entries[name]["measured_at"])]
            if self.store is not None:
                self.store.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", dir=self.store.parent, prefix=".choices-",
                                                     encoding="utf-8", delete=False) as output:
                        temporary = Path(output.name)
                        json.dump({"schema_version": 1, "entries": entries}, output, indent=2)
                        output.flush()
                        os.fsync(output.fileno())
                    temporary.replace(self.store)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            self.entries = entries


class AutoTuner:
    def __init__(self, runtime, *, metadata: Callable[[Path], dict], compatible: Callable[[dict], bool],
                 layout: Callable[[dict], dict], memory: Callable[[], tuple[int, int]],
                 hardware: Callable[[], dict], cpu_threads: Callable[[], int], cpu_count: Callable[[], int],
                 runtime_error: type[Exception], maximum: int = 262144, store: Path | None = None,
                 choices: EngineChoices | None = None, machine: Callable[[], dict | None] | None = None):
        self.runtime = runtime
        self.metadata = metadata
        self.compatible = compatible
        self.layout = layout
        self.memory = memory
        self.hardware = hardware
        self.cpu_threads = cpu_threads
        self.cpu_count = cpu_count
        self.runtime_error = runtime_error
        self.maximum = maximum
        self.store = store
        self.choices = choices
        self.machine = machine
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.worker: threading.Thread | None = None
        self.closed = False
        self.started: float | None = None
        self.report = {"state": "idle", "active": False, "trials": [], "recommendation": None, "error": None}
        if store is not None and store.exists():
            try:
                if store.stat().st_size > 1024 * 1024:
                    raise ValueError("Saved tuning report exceeds 1 MiB.")
                saved = json.loads(store.read_text(encoding="utf-8"))
                if (not isinstance(saved, dict) or saved.get("schema_version") != 1 or
                        saved.get("state") not in ("completed", "cancelled", "error") or
                        not isinstance(saved.get("trials"), list) or len(saved["trials"]) > 32 or
                        any(not isinstance(trial, dict) or not _valid_settings(trial.get("settings")) or
                            type(trial.get("capacity_passed")) is not bool or
                            (trial["capacity_passed"] and
                             (type(trial.get("prompt_tokens")) is not int or
                              type(trial.get("request_seconds")) not in (int, float)))
                            for trial in saved["trials"])):
                    raise ValueError("Saved tuning report has an unsupported format.")
                recommendation = saved.get("recommendation")
                if recommendation is not None:
                    if not isinstance(recommendation, dict) or not _valid_settings(recommendation.get("settings")):
                        raise ValueError("Saved tuning recommendation has invalid model settings.")
                    apps = recommendation.get("apps")
                    if (not isinstance(apps, dict) or
                            any(type(apps.get(key)) is not int or apps[key] <= 0 for key in
                                ("context", "maxInputTokens", "maxOutputTokens", "safetyMarginTokens")) or
                            apps["context"] != recommendation["settings"]["context"] or
                            apps["maxInputTokens"] + apps["maxOutputTokens"] + apps["safetyMarginTokens"] != apps["context"] or
                            type(recommendation.get("prompt_tokens_tested")) is not int):
                        raise ValueError("Saved tuning recommendation has invalid app limits.")
                self.report = {**saved, "active": False}
            except (OSError, ValueError) as exc:
                LOG.error("Saved auto-tuning report unavailable: %s", exc)
                self.report["error"] = f"Saved tuning report unavailable: {exc}"

    @property
    def active(self) -> bool:
        with self.lock:
            return self.report["active"]

    def snapshot(self) -> dict:
        with self.lock:
            result = deepcopy(self.report)
            if result["active"] and self.started is not None:
                result["elapsed_seconds"] = time.monotonic() - self.started
            return result

    def _update(self, **fields) -> None:
        with self.lock:
            self.report.update(fields)

    def start(self, data: dict) -> dict:
        if set(data) - {"model", "goal", "include_hybrid", "placement", "kv_cache_type", "confirm"}:
            raise ValueError("Auto Tune accepts model, goal, include_hybrid, placement, kv_cache_type and confirm.")
        if data.get("confirm") is not True:
            raise ValueError("Confirm that Auto Tune may unload the model and run local benchmarks.")
        goal = data.get("goal", "balanced")
        hybrid = data.get("include_hybrid")
        placement = data.get("placement", "auto")
        cache = data.get("kv_cache_type", "f16")
        if goal not in ("balanced", "maximum") or hybrid not in (None, True, False) or placement not in ("auto", "cpu"):
            raise ValueError("Select a supported tuning goal, placement and hybrid option.")
        model = data.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Choose a local GGUF model before tuning.")
        runtime = self.runtime()
        current = runtime.status()
        if cache not in ("f16", "q8_0") or (cache != "f16" and current["backend_kind"] != "llama.cpp"):
            raise ValueError("Select a supported KV cache precision; Q8 requires the official llama.cpp backend.")
        if self.closed or runtime.closed or self.active or current["chat_busy"] or current["state"] in ("loading", "stopping"):
            raise RuntimeError("Wait for the current model operation or chat to finish before tuning.")
        metadata = self.metadata(Path(model))
        architecture = metadata.get("general.architecture")
        limit = metadata.get(f"{architecture}.context_length", self.maximum)
        if type(limit) is not int or limit < 512:
            raise ValueError("The model has no usable context limit.")
        hybrid_possible = current["hybrid_available"] and placement == "auto" and self.compatible(metadata)
        if hybrid and not hybrid_possible:
            raise ValueError("Hybrid trials require an installed compatible MoE profile and automatic placement.")
        # By default every engine this installation supports for the model is measured.
        hybrid = hybrid_possible if hybrid is None else hybrid
        total, available = self.memory()
        if available < min(4 * 1024 ** 3, total // 10):
            raise RuntimeError("Insufficient available RAM safety reserve. Free memory before starting Auto Tune.")
        hardware = self.hardware()
        maximum = min(limit, self.maximum, 32768 if goal == "balanced" else self.maximum)
        contexts = [size for size in (8192, 16384, 32768, 65536, 131072, 262144) if size <= maximum]
        if not contexts or contexts[-1] != maximum:
            contexts.append(maximum)
        previous = {key: current[key] for key in LOAD_FIELDS} if current["state"] == "ready" else None
        with self.lock:
            self.cancelled.clear()
            self.started = time.monotonic()
            self.report = {
                "schema_version": 1, "state": "running", "active": True, "model": model, "goal": goal,
                "include_hybrid": hybrid, "phase": "Preparing local trials", "detail": "",
                "kv_cache_type": cache,
                "progress": None, "trials": [], "recommendation": None, "error": None,
                "hardware": {**hardware, "logical_cpus": self.cpu_count(), "ram_total_bytes": total,
                             "ram_available_at_start_bytes": available},
                "scope": ("Near-capacity input, short-answer and fixed-length generation checks; not a general "
                          "quality or sustained-speed qualification."),
            }
            self.worker = threading.Thread(
                target=self._run, args=(runtime, model, contexts, placement, hybrid, metadata, previous, cache, goal),
                daemon=True)
            self.worker.start()
        return self.snapshot()

    def cancel(self) -> dict:
        with self.lock:
            if self.report["active"]:
                self.cancelled.set()
                self.report["phase"] = "Cancelling; restoring the previous model"
        return self.snapshot()

    def close(self) -> None:
        self.closed = True
        self.cancelled.set()
        if self.worker is not None and self.worker is not threading.current_thread():
            self.worker.join(timeout=8)

    def _check(self, runtime) -> None:
        if self.cancelled.is_set() or self.closed or runtime.closed:
            raise Cancelled("Auto Tune was cancelled.")

    def _wait_ready(self, runtime, *, restoring: bool = False) -> dict:
        deadline = time.monotonic() + runtime.load_timeout + 10
        while time.monotonic() < deadline:
            if not restoring:
                self._check(runtime)
            elif self.closed or runtime.closed:
                raise Cancelled("The application is shutting down.")
            current = runtime.status()
            if current["state"] == "error":
                details = [line for line in current["logs"] if any(term in line.lower()
                           for term in ("failed", "out of memory", "error", "headroom"))][-3:]
                raise TrialError((current["error"] or "The backend could not load the trial.") +
                                 (" " + " ".join(details) if details else ""))
            if current["state"] == "ready":
                return current
            if current["state"] == "stopped":
                raise TrialError("The model was stopped before the trial became ready.")
            if self.cancelled.wait(0.25) and restoring:
                threading.Event().wait(0.1)
        raise TrialError("The backend did not become ready before its loading deadline.")

    def _native(self, runtime, path: str, data: dict, timeout: float = 30) -> dict:
        self._check(runtime)
        connection, key = runtime.connection()
        connection.timeout = timeout
        try:
            connection.request("POST", path, json.dumps(data),
                               {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            response = connection.getresponse()
            body = response.read(8 * 1024 * 1024 + 1)
            if response.status != 200 or len(body) > 8 * 1024 * 1024:
                raise TrialError(f"Backend {path} failed (HTTP {response.status}): {body[:300].decode(errors='replace')}")
            result = json.loads(body)
            if not isinstance(result, dict):
                raise TrialError(f"Backend {path} did not return a JSON object.")
            return result
        finally:
            connection.close()

    def _prompt(self, runtime, context: int) -> tuple[str, int]:
        target = context - max(128, min(8192, context // 8))
        before = ("SYNAPSE_TUNING_CHECK\nReturn ONLY JSON with keys start, middle, end, containing "
                  "the corresponding ANCHOR values. Ignore ordinary rows.\nANCHOR start = cedar4821\n")
        middle = "\nANCHOR middle = violet7319\n"
        after = "\nANCHOR end = quartz9067\nReturn the three anchor values as JSON. /no_think"

        def tokenize(text: str) -> list:
            result = self._native(runtime, "/tokenize", {"content": text, "add_special": False})
            tokens = result.get("tokens")
            if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
                raise TrialError("The backend tokenizer did not return integer tokens.")
            return tokens

        overhead = len(tokenize(before + middle + after))
        budget = target - overhead - 16
        if budget < 16:
            raise TrialError("The context is too small for the tuning check and output headroom.")
        filler = "".join(f"Row {index:06d}: item {index * 7919 % 1000003:07d}; state amber birch silver.\n"
                         for index in range(target // 12 + 1024))
        tokens = tokenize(filler)
        if len(tokens) < budget:
            raise TrialError("The tokenizer produced insufficient test input; no capacity claim was made.")
        parts = []
        for piece in (tokens[:budget // 2], tokens[budget // 2:budget]):
            decoded = self._native(runtime, "/detokenize", {"tokens": piece}).get("content")
            if not isinstance(decoded, str):
                raise TrialError("The backend detokenizer did not return text.")
            parts.append(decoded)
        prompt = before + parts[0] + middle + parts[1] + after
        count = len(tokenize(prompt))
        if not target - 128 <= count <= target + 32:
            raise TrialError(f"Test input has {count} tokens instead of approximately {target}.")
        return prompt, count

    def _probe(self, runtime, context: int) -> dict:
        prompt, count = self._prompt(runtime, context)
        self._check(runtime)
        connection, key = runtime.connection()
        connection.timeout = 2700
        stop = threading.Event()
        guard_errors = []
        stats = {"raw_prompt_tokens": count, "peak_rss_bytes": 0}
        started = time.monotonic()
        serial = runtime.begin_generation()

        def monitor():
            while not stop.wait(0.5):
                try:
                    self._check(runtime)
                    total, available = self.memory()
                    if available < min(4 * 1024 ** 3, total // 10):
                        raise TrialError("RAM safety reserve reached; the trial was stopped.")
                    current = runtime.status()
                    resources = current.get("resources") or {}
                    stats["peak_rss_bytes"] = max(stats["peak_rss_bytes"], resources.get("rss_bytes") or 0)
                    lines = [line for line in current["logs"] if "prompt processing" in line]
                    if lines:
                        match = re.search(r"progress\s*=\s*(0?\.\d+|1(?:\.0+)?)", lines[-1])
                        self._update(detail=lines[-1][-300:], progress=float(match[1]) if match else None)
                except (Cancelled, TrialError, OSError, ValueError) as exc:
                    guard_errors.append(exc)
                    if connection.sock is not None:
                        try:
                            connection.sock.shutdown(socket.SHUT_RDWR)
                        except OSError as shutdown_error:
                            LOG.info("Trial socket already closed during cancellation: %s", shutdown_error)
                    return

        thread = threading.Thread(target=monitor, daemon=True)
        result = []
        done = False
        thread.start()
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps({
                "model": "local-model", "messages": [{"role": "user", "content": prompt}],
                "stream": True, "max_tokens": 128, "temperature": 0, "seed": 42, "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True},
            }), {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            response = connection.getresponse()
            if response.status != 200:
                raise TrialError(f"Trial HTTP {response.status}: {response.read(1000).decode(errors='replace')}")
            for line in response:
                self._check(runtime)
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if not isinstance(event, dict):
                    raise TrialError("The backend returned a malformed stream event.")
                if event.get("error"):
                    raise TrialError(f"Backend trial error: {event['error']}")
                runtime.observe_generation(serial, event)
                if event.get("usage"):
                    stats["usage"] = event["usage"]
                if event.get("timings"):
                    stats["timings"] = event["timings"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        stats.setdefault("first_token_seconds", time.monotonic() - started)
                        result.append(delta["content"])
                    if choice.get("finish_reason"):
                        stats["finish_reason"] = choice["finish_reason"]
            if guard_errors:
                raise guard_errors[0]
            if not done:
                raise TrialError("The backend stream ended without its completion marker.")
            text = "".join(result).strip().removeprefix("```json").removesuffix("```").strip()
            try:
                answer = json.loads(text)
            except ValueError as exc:
                raise TrialError("The probe did not return the requested short JSON answer.") from exc
            if answer != ANCHORS:
                raise TrialError("The probe did not retrieve all three anchors correctly.")
            usage = stats.get("usage")
            actual = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            if type(actual) is not int or not count - 32 <= actual <= context - 64:
                raise TrialError(f"Backend usage did not verify the required input size: {actual}.")
            stats.update(prompt_tokens=actual, request_seconds=time.monotonic() - started,
                         anchors_passed=True, capacity_passed=True)
            return stats
        except (OSError, http.client.HTTPException):
            if guard_errors:
                raise guard_errors[0]
            raise
        finally:
            stop.set()
            connection.close()
            thread.join(timeout=2)

    def _decode(self, runtime) -> dict:
        """Steady generation speed after a short prompt: the speed of ordinary chat replies.

        Adaptive engines keep moving hot experts onto the GPU during the first few hundred tokens, so fixed-length
        chunks are generated until two consecutive chunks agree within 5% (at most six chunks)."""
        rates = []
        for _ in range(DECODE_MAX_CHUNKS):
            self._update(detail=f"Generation chunk {len(rates) + 1}" +
                         (f"; last {rates[-1]:.1f} tok/s" if rates else ""))
            result = self._native(runtime, "/completion", {
                "prompt": DECODE_PROMPT, "n_predict": DECODE_CHUNK_TOKENS, "ignore_eos": True, "temperature": 0,
                "seed": 42, "cache_prompt": False}, timeout=900)
            timings = result.get("timings")
            generated = timings.get("predicted_n") if isinstance(timings, dict) else None
            rate = timings.get("predicted_per_second") if isinstance(timings, dict) else None
            if (type(generated) is not int or generated < DECODE_CHUNK_TOKENS // 2 or
                    type(rate) not in (int, float) or rate <= 0):
                raise TrialError("The backend did not report a valid fixed-length generation speed.")
            rates.append(float(rate))
            if len(rates) >= 2 and abs(rates[-1] - rates[-2]) <= 0.05 * rates[-2]:
                break
        return {"decode_tokens_per_second": rates[-1], "decode_samples": rates,
                "decode_steady": len(rates) >= 2 and abs(rates[-1] - rates[-2]) <= 0.05 * rates[-2]}

    def _trial(self, runtime, settings: dict) -> dict:
        self._check(runtime)
        trial = {"settings": settings, "capacity_passed": False}
        label = f"{settings['context'] // 1024}K {settings['execution_mode']}"
        self._update(phase=f"Loading {label}", detail="", progress=None)
        try:
            runtime.stop()
            runtime.load(settings)
            loaded = self._wait_ready(runtime)
            if loaded["context"] != settings["context"]:
                raise TrialError(f"Backend loaded {loaded['context']} rather than {settings['context']} tokens.")
            self._update(phase=f"Testing near-capacity input: {label}", detail="", progress=None)
            if not runtime.chat_lock.acquire(blocking=False):
                raise TrialError("Another request acquired the model during the tuning trial.")
            try:
                probe = self._probe(runtime, settings["context"])
                timings = probe.get("timings") if isinstance(probe.get("timings"), dict) else {}
                for source, target in (("prompt_per_second", "prompt_tokens_per_second"),
                                       ("predicted_per_second", "long_context_decode_tokens_per_second")):
                    if type(timings.get(source)) in (int, float) and timings[source] > 0:
                        probe[target] = float(timings[source])
                self._update(phase=f"Measuring generation speed: {label}", detail="", progress=None)
                probe.update(self._decode(runtime))
                trial.update(probe)
            finally:
                runtime.chat_lock.release()
        except (self.runtime_error, TrialError, OSError, ValueError, http.client.HTTPException) as exc:
            if isinstance(exc, Cancelled) or self.cancelled.is_set():
                raise Cancelled("Auto Tune was cancelled.") from exc
            trial["error"] = str(exc)
            LOG.warning("Auto Tune %s trial failed: %s", label, exc)
        with self.lock:
            self.report["trials"].append(trial)
        return trial

    @staticmethod
    def _better(candidate: dict, incumbent: dict, goal: str) -> bool:
        # Balanced favors everyday reply speed; Maximum favors long-prompt throughput. Both need a 5% margin.
        if goal == "balanced":
            return candidate["decode_tokens_per_second"] >= incumbent["decode_tokens_per_second"] * 1.05
        return candidate["request_seconds"] <= incumbent["request_seconds"] * 0.95

    def _remember(self, model: str, selected: dict, standard: dict, goal: str) -> bool:
        machine = self.machine() if self.machine is not None else None
        if self.choices is None or machine is None:
            return False
        settings = {key: selected["settings"][key] for key in CHOICE_FIELDS}
        try:
            self.choices.remember(Path(model), machine, {
                "settings": settings, "goal": goal, "measured_at": time.time(),
                "decode_tokens_per_second": selected["decode_tokens_per_second"],
                "standard_decode_tokens_per_second": standard["decode_tokens_per_second"]})
        except (OSError, ValueError) as exc:
            LOG.error("Could not remember the measured engine choice: %s", exc)
            return False
        return True

    def _apply(self, runtime, settings: dict | None, budget: int, *, restoring: bool,
               expected: dict | None = None) -> None:
        with runtime.lock:
            runtime.hybrid_context = budget
        runtime.stop()
        if settings is not None:
            total, available = self.memory()
            if available < min(4 * 1024 ** 3, total // 10):
                raise TrialError("RAM safety reserve prevents reloading the model; it has been left unloaded.")
            runtime.load(settings)
            loaded = self._wait_ready(runtime, restoring=restoring)
            measured = expected is not None
            expected = expected or settings
            if loaded["context"] != expected["context"]:
                raise TrialError(f"Applied context {loaded['context']} differs from the selected {expected['context']}.")
            if measured and (loaded["hybrid_active"] != (expected["execution_mode"] == "hybrid") or
                                                 (loaded["hybrid_active"] and
                                                  loaded["hybrid_variant"] != expected["hybrid_variant"])):
                raise TrialError("The applied engine differs from the measured choice.")

    def _run(self, runtime, model: str, contexts: list[int], placement: str, hybrid: bool,
             metadata: dict, previous: dict | None, cache: str, goal: str = "balanced") -> None:
        original_budget = runtime.hybrid_context
        selected = None
        remembered = False
        final_state = "completed"
        error = None
        try:
            for context in contexts:
                settings = {"model": model, "execution_mode": "standard", "context": context,
                            "placement": placement, "cpu_threads": min(8, self.cpu_threads()),
                            "batch_threads": min(16, self.cpu_count()), "fit_target_mib": 1024,
                            "kv_cache_type": cache, "expert_cache_mib": 0, "hybrid_variant": "classic"}
                trial = self._trial(runtime, settings)
                if not trial["capacity_passed"]:
                    break
                selected = trial
            if selected is None:
                raise TrialError("No context passed. Review the failed trials; no app settings were recommended.")
            standard = selected
            if hybrid:
                context = selected["settings"]["context"]
                bound = runtime.hybrid_plan != "auto" and runtime.hybrid_layout == self.layout(metadata)
                baseline = runtime.hybrid_cache_mib if bound else runtime.hybrid_generic_cache_mib
                banks = [baseline] if bound else sorted({min(baseline, size) for size in
                                                        (baseline, 6144, 4096, 2048, 1024)}, reverse=True)
                transfer = runtime.hybrid_transfer if bound else runtime.hybrid_generic_transfer
                with runtime.lock:
                    runtime.hybrid_context = context
                best_hybrid = None
                for bank in banks:
                    settings = {"model": model, "execution_mode": "hybrid", "context": context,
                                "placement": "auto", "cpu_threads": min(8, self.cpu_threads()),
                                "batch_threads": min(16, self.cpu_count()), "expert_cache_mib": bank,
                                "load_mode": "auto", "host_op_offload": False, "fit_target_mib": 1024,
                                "kv_cache_type": cache, "hybrid_variant": "classic"}
                    trial = self._trial(runtime, settings)
                    if trial["capacity_passed"]:
                        best_hybrid = trial
                        if runtime.hosted_capable and transfer == "direct":
                            hosted = self._trial(runtime, {**settings, "hybrid_variant": "hosted"})
                            if hosted["capacity_passed"] and self._better(hosted, best_hybrid, goal):
                                best_hybrid = hosted
                        break
                    if not any(term in trial.get("error", "").lower() for term in
                               ("allocat", "out of memory", "cuda", "residency", "cache")):
                        break
                if best_hybrid is not None and self._better(best_hybrid, selected, goal):
                    selected = best_hybrid
            remembered = self._remember(model, selected, standard, goal)
        except Cancelled as exc:
            final_state, error = "cancelled", str(exc)
            selected = None
        except (self.runtime_error, TrialError, OSError, ValueError, http.client.HTTPException) as exc:
            final_state, error = "error", str(exc)
            selected = None
            LOG.error("Auto Tune failed: %s", exc)
        finally:
            self._update(phase="Applying measured settings" if selected else "Restoring the previous model",
                         detail="", progress=None)
            if not self.closed and not runtime.closed:
                try:
                    settings = selected["settings"] if selected else previous
                    budget = (settings["context"] if selected and settings["execution_mode"] == "hybrid"
                              and not remembered else original_budget)
                    if remembered:
                        # Load through Automatic so the remembered choice is what users get from now on.
                        self._apply(runtime, {"model": model, "execution_mode": "auto", "kv_cache_type": cache},
                                    budget, restoring=False, expected=settings)
                    else:
                        self._apply(runtime, settings, budget, restoring=selected is None)
                except Cancelled as exc:
                    final_state, error, selected = "cancelled", str(exc), None
                    if not self.closed and not runtime.closed:
                        try:
                            self._apply(runtime, previous, original_budget, restoring=True)
                        except (self.runtime_error, TrialError, Cancelled, OSError, ValueError) as restore_error:
                            final_state, error = "error", f"Cancellation could not restore the model: {restore_error}"
                            LOG.error("%s", error)
                except (self.runtime_error, TrialError, OSError, ValueError) as exc:
                    final_state = "error"
                    error = f"{error + ' ' if error else ''}Could not apply or restore the model: {exc}"
                    restore = selected is not None
                    selected = None
                    LOG.error("%s", error)
                    if restore:
                        try:
                            self._apply(runtime, previous, original_budget, restoring=True)
                        except (self.runtime_error, TrialError, Cancelled, OSError, ValueError) as restore_error:
                            error += f" Restoring the previous model also failed: {restore_error}"
                            LOG.error("%s", error)
            recommendation = None
            if selected is not None and not self.closed:
                recommendation = {"settings": selected["settings"],
                                  "apps": client_settings(selected["settings"]["context"]),
                                  "prompt_tokens_tested": selected["prompt_tokens"],
                                  "request_seconds": selected["request_seconds"],
                                  "decode_tokens_per_second": selected["decode_tokens_per_second"],
                                  "selection_metric": ("generation speed" if goal == "balanced"
                                                       else "long-prompt request time")}
            with self.lock:
                report = deepcopy(self.report)
            report.update(state=final_state, active=False, error=error, recommendation=recommendation,
                          phase="Complete" if final_state == "completed" else final_state.title(),
                          elapsed_seconds=time.monotonic() - self.started, completed_at=time.time())
            if self.store is not None and not self.closed:
                temporary = None
                try:
                    self.store.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    with tempfile.NamedTemporaryFile(mode="w", dir=self.store.parent, prefix=".tuning-",
                                                     encoding="utf-8", delete=False) as output:
                        temporary = Path(output.name)
                        json.dump(report, output, indent=2)
                        output.flush()
                        os.fsync(output.fileno())
                    temporary.replace(self.store)
                except (OSError, ValueError) as exc:
                    LOG.error("Could not save tuning results: %s", exc)
                    report["error"] = f"Could not save tuning results: {exc}"
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            with self.lock:
                self.report = report
