#!/usr/bin/env python3

import http.client
import importlib.util
import io
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from unittest import mock
from urllib.parse import urlsplit


APP_PATH = Path(sys.argv.pop(1)).resolve()
spec = importlib.util.spec_from_file_location("vrambridge_app", APP_PATH)
assert spec is not None and spec.loader is not None
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

HYBRID_METADATA = {
    "general.architecture": "gpt-oss", "general.file_type": 38,
    "gpt-oss.block_count": 36, "gpt-oss.embedding_length": 2880,
    "gpt-oss.feed_forward_length": 2880, "gpt-oss.expert_count": 128,
    "gpt-oss.expert_used_count": 4,
}

MTP_METADATA = {
    "general.architecture": "qwen4exp", "general.file_type": 2,
    "qwen4exp.block_count": 2, "qwen4exp.embedding_length": 2560,
    "qwen4exp.expert_count": 512, "qwen4exp.expert_used_count": 10,
    "qwen4exp.expert_feed_forward_length": 640, "qwen4exp.vocab_size": 151936,
}


def write_metadata(path, metadata):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    entries = []
    for key, value in metadata.items():
        if isinstance(value, list):
            kind, raw = 9, struct.pack("<IQ", 4, len(value)) + b"".join(struct.pack("<I", item) for item in value)
        else:
            kind, raw = (8, string(value)) if isinstance(value, str) else (4, struct.pack("<I", value))
        entries.append(string(key) + struct.pack("<I", kind) + raw)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries)) + b"".join(entries))


class LocalAppTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.backend = self.root / "llama-server.py"
        shutil.copyfile(Path(__file__).with_name("fake_llama_server.py"), self.backend)
        self.backend.chmod(0o755)
        self.model = self.root / "model with spaces.gguf"
        self.model.write_bytes(b"GGUF-test-model")
        self.runtime = app.Runtime(str(self.backend))
        self.server = app.AppServer(("127.0.0.1", 0), self.runtime)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.runtime.stop(closing=True)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temporary.cleanup()

    def request(self, method, path, data=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            supplied = {"Content-Type": "application/json"}
            supplied.update(headers or {})
            body = json.dumps(data) if data is not None else None
            connection.request(method, path, body=body, headers=supplied)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def wait_state(self, wanted):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = self.runtime.status()
            if status["state"] == wanted:
                return status
            time.sleep(0.03)
        self.fail(f"Expected {wanted}: {self.runtime.status()}")

    def load(self, model=None, **extra):
        code, _, body = self.request("POST", "/api/v1/model/load",
                                    {"model": str(model or self.model), **extra})
        self.assertEqual(code, 202, body)
        self.assertEqual(json.loads(body)["state"], "loading")
        return self.wait_state("ready")

    def wait_tuning(self, wanted):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            result = self.server.tuner.snapshot()
            if not result["active"] and result["state"] == wanted:
                return result
            time.sleep(0.03)
        self.fail(f"Expected tuning {wanted}: {self.server.tuner.snapshot()}")

    def test_static_assets_status_and_unknown_routes(self):
        for path, expected in (("/", b"SYNAPSE"), ("/app.js", b"textContent"),
                               ("/chat-stream.mjs", b"class ChatStream"),
                               ("/chat-input.mjs", b"bindChatInput"), ("/theme.js", b"synapse.theme"),
                               ("/model-controls.mjs", b"bindReasoningEffort"),
                               ("/monitor.mjs", b"class MonitorView"), ("/navigation.mjs", b"viewForHash"),
                               ("/updates.mjs", b"class UpdateView"),
                               ("/icon.svg", b"<svg"),
                               ("/style.css", b"@media")):
            code, headers, body = self.request("GET", path)
            self.assertEqual(code, 200)
            self.assertIn(expected, body)
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
            self.assertEqual(headers["Cache-Control"], "no-store")
        code, _, body = self.request("GET", "/api/v1/status")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["state"], "stopped")
        for path in ("/../app.py", "/%2e%2e/app.py", "/api/v2/status"):
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_loopback_startup_does_not_require_reverse_dns(self):
        runtime = app.Runtime(str(self.backend))
        with mock.patch.object(app.socket, "getfqdn", side_effect=AssertionError("Unexpected DNS lookup")):
            server = app.AppServer(("127.0.0.1", 0), runtime)
        try:
            self.assertEqual(server.server_name, "localhost")
            self.assertEqual(server.server_port, server.server_address[1])
        finally:
            server.server_close()
            runtime.stop(closing=True)

    def test_missing_backend_is_explicit(self):
        self.runtime.executable = None
        self.assertFalse(self.runtime.status()["backend_available"])
        code, _, body = self.request("POST", "/api/v1/model/load", {"model": str(self.model)})
        self.assertEqual(code, 503)
        self.assertIn(b"--llama-server", body)

    def test_update_status_and_installation_guards(self):
        self.server.updates.automatic = False
        code, _, body = self.request("GET", "/api/v1/update")
        self.assertEqual(code, 200)
        self.assertFalse(json.loads(body)["managed"])
        with mock.patch.object(self.server.updates, "check") as check:
            self.assertEqual(self.request("POST", "/api/v1/update/check", {})[0], 202)
            check.assert_called_once_with(force=True)
        self.assertEqual(self.request("POST", "/api/v1/update/apply", {"version": "1.0.0"})[0], 400)
        self.runtime.chat_lock.acquire()
        try:
            self.assertEqual(self.request("POST", "/api/v1/update/apply", {"version": "1.0.0"})[0], 409)
        finally:
            self.runtime.chat_lock.release()
        self.assertEqual(self.request("POST", "/api/v1/update/check", {},
                                     {"Origin": "https://untrusted.invalid"})[0], 403)

    def test_managed_update_checks_artifact_and_requests_an_idle_restart(self):
        archive = self.root / "SYNAPSE.pyz"
        archive.write_bytes(b"previous app")
        (self.root / "install.json").write_text(json.dumps(
            {"schema_version": 1, "app": "SYNAPSE", "mode": "managed"}))
        self.server.updates = app._updates.UpdateManager("0.1.0rc1", archive, automatic=False)
        self.server.updates.state = "available"
        self.server.updates.release = {"tag": "v0.2.0rc1", "version": "0.2.0rc1"}
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as output:
            output.writestr("__main__.py", "")
            output.writestr("synapse_app/app.py", "")
            output.writestr("synapse_app/__init__.py", '__version__ = "0.2.0rc1"')
        body = payload.getvalue()
        checksum = (hashlib.sha256(body).hexdigest() + "  SYNAPSE.pyz\n").encode()
        with mock.patch.object(app._updates, "download", side_effect=[checksum, body]):
            code, _, response = self.request("POST", "/api/v1/update/apply", {"version": "0.2.0rc1"})
            self.assertEqual(code, 202, response)
            self.thread.join(timeout=5)
        self.assertFalse(self.thread.is_alive())
        self.assertTrue(self.server.restart_requested)
        self.assertEqual(app._updates.validate_archive(archive), "0.2.0rc1")
        self.assertEqual(archive.with_suffix(".pyz.previous").read_bytes(), b"previous app")

    def test_update_restart_never_interrupts_a_chat_that_started_during_download(self):
        self.runtime.chat_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "chat started"):
                self.server.install_update(self.root / "unused.pyz", {})
            self.assertFalse(self.server.restart_requested)
            self.assertFalse(self.runtime.closed)
        finally:
            self.runtime.chat_lock.release()

    def test_model_browser_only_lists_folders_and_gguf_files(self):
        (self.root / "nested folder").mkdir()
        (self.root / "ignored.txt").write_text("private contents must not be returned")
        (self.root / ".hidden.gguf").write_bytes(b"GGUF")
        code, _, body = self.request("POST", "/api/v1/models/browse", {"path": str(self.root)})
        self.assertEqual(code, 200, body)
        data = json.loads(body)
        self.assertEqual(data["path"], str(self.root))
        self.assertEqual([e["name"] for e in data["entries"]], ["nested folder", self.model.name])
        self.assertTrue(data["entries"][0]["directory"])
        self.assertFalse(data["entries"][1]["directory"])
        self.assertNotIn(b"private contents", body)
        self.assertNotIn(b"GGUF-test-model", body)
        self.assertFalse(data["truncated"])
        self.assertIsNone(self.runtime.process)
        code, _, body = self.request("POST", "/api/v1/models/browse", {"path": str(self.model)})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["path"], str(self.root))

    def test_model_browser_validates_paths_permissions_and_request_guard(self):
        for value in ("", "relative/folder", True, 5, [], "x" * 4097):
            with self.subTest(value=str(value)[:30]):
                self.assertEqual(self.request("POST", "/api/v1/models/browse", {"path": value})[0], 400)
        self.assertEqual(self.request("POST", "/api/v1/models/browse", {"unknown": 1})[0], 400)
        self.assertEqual(self.request("POST", "/api/v1/models/browse", {"path": str(self.root / "absent")})[0], 404)
        with mock.patch.object(app.os, "scandir", side_effect=PermissionError("denied")):
            self.assertEqual(self.request("POST", "/api/v1/models/browse", {"path": str(self.root)})[0], 403)
        with mock.patch.object(Path, "home", return_value=self.root):
            code, _, body = self.request("POST", "/api/v1/models/browse", {})
            self.assertEqual(code, 200)
            self.assertEqual(json.loads(body)["path"], str(self.root))
        self.assertEqual(self.request("POST", "/api/v1/models/browse", {},
                                     {"Origin": "https://untrusted.invalid"})[0], 403)
        root = Path(self.root.anchor)
        with mock.patch.object(app.os, "scandir") as scan:
            scan.return_value.__enter__.return_value = iter([])
            self.assertIsNone(app.browse_models(str(root))["parent"])

    def test_model_browser_listing_is_bounded_and_surfaces_truncation(self):
        entry = mock.Mock(name="entry")
        entry.name = "model.gguf"
        entry.is_dir.return_value = False
        entry.is_file.return_value = True
        with mock.patch.object(app.os, "scandir") as scan:
            scan.return_value.__enter__.return_value = iter([entry] * 5000)
            data = app.browse_models(str(self.root))
        self.assertEqual(len(data["entries"]), 512)
        self.assertTrue(data["truncated"])
        self.assertEqual(entry.is_dir.call_count, 4096)

    def test_load_models_api_and_unload(self):
        status = self.load()
        self.assertEqual(status["model"], str(self.model))
        self.assertEqual(status["placement"], "auto")
        code, _, body = self.request("GET", "/v1/models")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], "local-model")
        self.assertEqual(self.request("POST", "/api/v1/model/load", {"model": str(self.model)})[0], 409)
        process = self.runtime.process
        self.assertIsNotNone(process)
        self.assertEqual(self.request("POST", "/api/v1/model/unload", {})[0], 200)
        self.assertIsNotNone(process.poll())
        self.assertEqual(self.runtime.status()["state"], "stopped")
        self.assertEqual(self.request("GET", "/v1/models")[0], 503)

    def test_standard_context_boundaries_and_ui_limit(self):
        code, _, body = self.request("GET", "/")
        self.assertEqual(code, 200)
        self.assertRegex(body.decode(), r'<input id="context"[^>]*max="262144"')
        for context in (512, 65536, 65537, 131072, 262144):
            with self.subTest(context=context):
                status = self.load(context=context, execution_mode="standard")
                self.assertEqual(status["requested_context"], context)
                self.assertEqual(status["context"], context)
                arguments = self.runtime.process.args
                self.assertEqual(arguments[arguments.index("--ctx-size") + 1], str(context))
                self.runtime.stop()

    def test_large_context_prompt_body_is_forwarded(self):
        self.load(context=262144)
        payload = {"model": "local-model", "messages": [{"role": "user", "content": "sample " * 220000}]}
        self.assertGreater(len(json.dumps(payload).encode()), 1024 * 1024)
        code, _, body = self.request("POST", "/v1/chat/completions", payload)
        self.assertEqual(code, 200, body)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "Hello")

    def test_loading_progress_and_cancellation_are_visible(self):
        model = self.root / "slow.gguf"
        model.write_bytes(b"GGUF")
        code, _, body = self.request("POST", "/api/v1/model/load", {"model": str(model)})
        self.assertEqual(code, 202)
        loading = json.loads(body)["loading"]
        self.assertIsNotNone(loading)
        self.assertGreaterEqual(loading["elapsed_seconds"], 0)
        self.assertEqual(loading["phase"], "Starting backend")
        self.assertEqual(self.request("POST", "/api/v1/model/unload", {})[0], 200)
        self.assertEqual(self.runtime.status()["state"], "stopped")
        self.assertIsNone(self.runtime.status()["loading"])
        self.assertIsNone(self.runtime.process)

    def test_hybrid_verification_has_visible_pending_checks_and_separate_deadline(self):
        self.hybrid_runtime(load_timeout=5)
        model = self.root / "hybrid-no-markers.gguf"
        write_metadata(model, HYBRID_METADATA)
        with mock.patch.object(app, "HYBRID_VERIFICATION_TIMEOUT", 0.5):
            started = time.monotonic()
            self.assertEqual(self.request("POST", "/api/v1/model/load", {"model": str(model)})[0], 202)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                current = self.runtime.status()
                if current["loading"] and current["loading"]["waiting_for"]:
                    break
                time.sleep(0.03)
            self.assertEqual(current["loading"]["phase"], "Verifying experimental hybrid execution")
            self.assertIn("hybrid dispatcher activation", current["loading"]["waiting_for"])
            failed = self.wait_state("error")
            self.assertIn("Unconfirmed", failed["error"])
            self.assertLess(time.monotonic() - started, 3)

    def test_auto_tune_balanced_measures_capacity_applies_and_saves_app_limits(self):
        write_metadata(self.model, {"general.architecture": "fixture", "fixture.context_length": 262144})
        self.server.tuner.store = self.root / "tuning.json"
        code, _, body = self.request("POST", "/api/v1/tuning/start",
                                    {"model": str(self.model), "confirm": True})
        self.assertEqual(code, 202, body)
        result = self.wait_tuning("completed")
        self.assertEqual([trial["settings"]["context"] for trial in result["trials"]], [8192, 16384, 32768])
        self.assertTrue(all(trial["capacity_passed"] for trial in result["trials"]))
        self.assertEqual(self.runtime.status()["context"], 32768)
        apps = result["recommendation"]["apps"]
        self.assertEqual(apps["maxInputTokens"], 24576)
        self.assertEqual(apps["maxOutputTokens"], 4096)
        self.assertLess(apps["maxInputTokens"] + apps["maxOutputTokens"], apps["context"])
        self.assertEqual(json.loads(self.server.tuner.store.read_text())["state"], "completed")
        self.assertNotIn("ANCHOR", self.server.tuner.store.read_text())
        code, _, body = self.request("GET", "/api/v1/status")
        self.assertEqual(json.loads(body)["tuning"]["state"], "completed")

    def test_auto_tune_maximum_respects_model_context_limit(self):
        write_metadata(self.model, {"general.architecture": "fixture", "fixture.context_length": 65536})
        code, _, body = self.request("POST", "/api/v1/tuning/start",
                                    {"model": str(self.model), "goal": "maximum", "confirm": True})
        self.assertEqual(code, 202, body)
        result = self.wait_tuning("completed")
        self.assertEqual(result["recommendation"]["settings"]["context"], 65536)
        self.assertEqual(max(trial["settings"]["context"] for trial in result["trials"]), 65536)

    def test_auto_tune_cancel_restores_model_and_excludes_other_operations(self):
        write_metadata(self.model, {"general.architecture": "fixture", "fixture.context_length": 32768})
        self.load(context=4096)
        entered = threading.Event()

        def probe(runtime, context):
            entered.set()
            self.server.tuner.cancelled.wait(5)
            raise app._tuning.Cancelled("Cancelled fixture probe.")

        with mock.patch.object(self.server.tuner, "_probe", side_effect=probe):
            self.assertEqual(self.request("POST", "/api/v1/tuning/start",
                                         {"model": str(self.model), "confirm": True})[0], 202)
            self.assertTrue(entered.wait(3))
            for path, payload in (("/api/v1/model/load", {"model": str(self.model)}),
                                  ("/api/v1/model/unload", {}),
                                  ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}),
                                  ("/api/v1/tuning/start", {"model": str(self.model), "confirm": True})):
                self.assertEqual(self.request("POST", path, payload)[0], 409, path)
            with self.assertRaisesRegex(RuntimeError, "Auto Tune"):
                self.server.install_update(self.root / "unused.pyz", {})
            self.assertEqual(self.request("POST", "/api/v1/tuning/cancel", {})[0], 202)
            self.assertEqual(self.wait_tuning("cancelled")["recommendation"], None)
        self.assertEqual(self.runtime.status()["state"], "ready")
        self.assertEqual(self.runtime.status()["context"], 4096)

    def test_auto_tune_failed_probe_restores_previous_model_without_recommendations(self):
        model = self.root / "tuning-wrong-answer.gguf"
        write_metadata(model, {"general.architecture": "fixture", "fixture.context_length": 32768})
        self.load(model=model, context=4096)
        self.assertEqual(self.request("POST", "/api/v1/tuning/start",
                                     {"model": str(model), "confirm": True})[0], 202)
        result = self.wait_tuning("error")
        self.assertIn("No context passed", result["error"])
        self.assertIn("anchors", result["trials"][0]["error"])
        self.assertIsNone(result["recommendation"])
        self.assertEqual(self.runtime.status()["context"], 4096)

    def test_auto_tune_validates_consent_goal_and_hybrid_opt_in(self):
        for changes in ({}, {"confirm": False}, {"confirm": True, "goal": "unknown"},
                        {"confirm": True, "include_hybrid": "yes"},
                        {"confirm": True, "include_hybrid": True}, {"confirm": True, "unknown": 1}):
            with self.subTest(changes=changes):
                self.assertEqual(self.request("POST", "/api/v1/tuning/start",
                                             {"model": str(self.model), **changes})[0], 400)
        self.assertFalse(self.server.tuner.active)
        self.runtime.chat_lock.acquire()
        try:
            self.assertEqual(self.request("POST", "/api/v1/tuning/start",
                                         {"model": str(self.model), "confirm": True})[0], 409)
        finally:
            self.runtime.chat_lock.release()
        self.assertEqual(self.request("POST", "/api/v1/tuning/cancel", {"unknown": True})[0], 400)

    def test_auto_tune_hybrid_trials_reduce_unbound_cache_without_changing_the_plan(self):
        self.hybrid_runtime(hybrid_generic=True, hybrid_context=8192, hybrid_cache_mib=8192)
        model = self.root / "tuning-cache-limit.gguf"
        metadata = {"general.architecture": "qwen2moe", "general.file_type": 7,
                    "qwen2moe.block_count": 3, "qwen2moe.embedding_length": 96,
                    "qwen2moe.feed_forward_length": 192, "qwen2moe.expert_count": 7,
                    "qwen2moe.expert_used_count": 3, "qwen2moe.context_length": 16384}
        write_metadata(model, metadata)
        original = self.plan.read_bytes()
        code, _, body = self.request("POST", "/api/v1/tuning/start",
                                    {"model": str(model), "confirm": True, "include_hybrid": True})
        self.assertEqual(code, 202, body)
        result = self.wait_tuning("completed")
        trials = [trial for trial in result["trials"] if trial["settings"]["execution_mode"] == "hybrid"]
        self.assertEqual([trial["settings"]["expert_cache_mib"] for trial in trials], [8192, 6144, 4096])
        self.assertFalse(trials[0]["capacity_passed"])
        self.assertFalse(trials[1]["capacity_passed"])
        self.assertTrue(trials[2]["capacity_passed"])
        self.assertEqual(self.plan.read_bytes(), original)
        self.assertEqual(self.runtime.status()["state"], "ready")
        self.assertEqual(self.runtime.status()["context"], 16384)

    def test_auto_tune_cancellation_while_applying_restores_previous_settings(self):
        write_metadata(self.model, {"general.architecture": "fixture", "fixture.context_length": 8192})
        self.load(context=4096)
        apply = self.server.tuner._apply

        def cancel_apply(runtime, settings, budget, *, restoring):
            if not restoring:
                self.server.tuner.cancel()
            return apply(runtime, settings, budget, restoring=restoring)

        with mock.patch.object(self.server.tuner, "_apply", side_effect=cancel_apply):
            self.assertEqual(self.request("POST", "/api/v1/tuning/start",
                                         {"model": str(self.model), "confirm": True})[0], 202)
            result = self.wait_tuning("cancelled")
        self.assertIsNone(result["recommendation"])
        self.assertEqual(self.runtime.status()["context"], 4096)

    def test_auto_tune_refuses_to_start_without_ram_headroom(self):
        write_metadata(self.model, {"general.architecture": "fixture", "fixture.context_length": 8192})
        with mock.patch.object(self.server.tuner, "memory", return_value=(16 * 1024**3, 1024**3)):
            code, _, body = self.request("POST", "/api/v1/tuning/start",
                                        {"model": str(self.model), "confirm": True})
        self.assertEqual(code, 409)
        self.assertIn(b"RAM safety reserve", body)
        self.assertFalse(self.server.tuner.active)
        self.assertIsNone(self.runtime.process)

    def test_cpu_flags_and_environment_isolation(self):
        with mock.patch.dict(os.environ, {"LLAMA_ARG_MODEL": "bad",
                                         "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",
                                         "VRAMBRIDGE_HYBRID_PLAN": "/unexpected/plan",
                                         "VRAMBRIDGE_HYBRID_DIRECT": "1",
                                         "GGML_CUDA_REGISTER_HOST": "1"}):
            status = self.load(placement="cpu")
        self.assertIn('"gpu_layers": "0"', "\n".join(status["logs"]))
        self.assertIn('"inherited_override": false', "\n".join(status["logs"]))
        self.assertIn('"managed_memory": false', "\n".join(status["logs"]))
        self.assertNotIn(self.runtime.key, "\n".join(status["logs"]))
        flags = json.loads(status["logs"][0])
        self.assertIsNone(flags["hybrid_plan"])
        self.assertIsNone(flags["hybrid_direct"])
        self.assertIsNone(flags["register_host"])
        self.assertEqual(status["execution_profile"], "standard")
        self.assertIsNone(status["execution_warning"])

    def test_auto_flags(self):
        status = self.load()
        self.assertIn('"gpu_layers": "auto"', "\n".join(status["logs"]))
        self.assertIn('"fit_target": "1024"', "\n".join(status["logs"]))

    def test_q8_cache_requires_explicit_selection_and_is_backend_verified(self):
        default = self.load()
        self.assertEqual(default["kv_cache_type"], "f16")
        arguments = self.runtime.process.args
        self.assertNotIn("--cache-type-k", arguments)
        self.assertNotIn("--cache-type-v", arguments)
        self.assertNotIn("--flash-attn", arguments)
        self.assertNotIn("--log-verbosity", arguments)
        self.runtime.stop()
        selected = self.load(kv_cache_type="q8_0")
        arguments = self.runtime.process.args
        self.assertEqual(arguments[arguments.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(arguments[arguments.index("--cache-type-v") + 1], "q8_0")
        self.assertEqual(arguments[arguments.index("--flash-attn") + 1], "on")
        self.assertEqual(arguments.count("--flash-attn"), 1)
        self.assertEqual(arguments[arguments.index("--log-verbosity") + 1], "4")
        self.assertTrue(selected["kv_cache_verified"])
        self.assertTrue(selected["flash_attention_verified"])
        self.assertEqual(selected["kv_cache_type"], "q8_0")
        retained_logs = "\n".join(selected["logs"])
        self.assertNotIn("api_keys:", retained_logs)
        self.assertNotIn("[fixture marker]", retained_logs)
        self.assertIn("Backend authentication configured.", selected["logs"])

    def test_q8_cache_rejects_invalid_values_and_unsupported_backends(self):
        for value in (None, True, 0, "q4_0", "unknown"):
            self.assertEqual(self.request("POST", "/api/v1/model/load",
                                         {"model": str(self.model), "kv_cache_type": value})[0], 400)
        self.runtime.backend_kind = "ik_llama.cpp"
        code, _, body = self.request("POST", "/api/v1/model/load",
                                    {"model": str(self.model), "kv_cache_type": "q8_0"})
        self.assertEqual(code, 400)
        self.assertIn(b"official llama.cpp", body)
        self.assertIsNone(self.runtime.process)

    def test_q8_cache_does_not_accept_a_silent_precision_fallback(self):
        model = self.root / "q8-cache-fallback.gguf"
        model.write_bytes(b"GGUF")
        self.assertEqual(self.request("POST", "/api/v1/model/load",
                                     {"model": str(model), "kv_cache_type": "q8_0"})[0], 202)
        failed = self.wait_state("error")
        self.assertIn("No precision fallback was accepted", failed["error"])
        self.assertFalse(failed["kv_cache_verified"])

    def test_auto_tune_preserves_explicit_q8_precision_in_trials_and_applied_result(self):
        write_metadata(self.model, {"general.architecture": "fixture", "fixture.context_length": 8192})
        self.assertEqual(self.request("POST", "/api/v1/tuning/start",
                                     {"model": str(self.model), "confirm": True, "kv_cache_type": "q8_0"})[0], 202)
        result = self.wait_tuning("completed")
        self.assertEqual(result["recommendation"]["settings"]["kv_cache_type"], "q8_0")
        self.assertTrue(all(trial["settings"]["kv_cache_type"] == "q8_0" for trial in result["trials"]))
        self.assertEqual(self.runtime.status()["kv_cache_type"], "q8_0")
        self.assertTrue(self.runtime.status()["kv_cache_verified"])

    def hybrid_runtime(self, **extra):
        self.plan = self.root / "hot experts.txt"
        self.plan.write_text("# training-only test plan\n0 7\n35 127\n")
        self.plugin = self.root / "libggml-cuda.so"
        self.plugin.write_bytes(b"test-plugin")
        write_metadata(self.model, HYBRID_METADATA)
        extra.setdefault("hybrid_layout", app.moe_layout(HYBRID_METADATA))
        self.runtime = app.Runtime(str(self.backend), hybrid_plan=self.plan,
                                   hybrid_cuda_plugin=self.plugin, **extra)
        self.server.runtime = self.runtime

    def test_hybrid_profile_flags_environment_and_readiness(self):
        self.hybrid_runtime()
        self.assertEqual(self.runtime.status()["context"], 16384)
        self.assertFalse(self.runtime.status()["hybrid_verified"])
        with mock.patch.dict(os.environ, {"VRAMBRIDGE_HYBRID_BATCH": "32",
                                         "VRAMBRIDGE_HYBRID_REPACK": "1",
                                         "VRAMBRIDGE_HYBRID_PLAN": "/unexpected/plan",
                                         "GGML_BACKEND_PATH": "/unexpected/plugin",
                                         "VRAMBRIDGE_HYBRID_OVERLAP": "1",
                                         "VRAMBRIDGE_HYBRID_FUSE_GATE_UP": "1",
                                         "GGML_CUDA_REGISTER_HOST": "0"}):
            status = self.load()
        flags = json.loads(status["logs"][0])
        self.assertEqual(status["execution_profile"], "experimental-hybrid-direct")
        self.assertTrue(status["hybrid_verified"])
        self.assertFalse(status["fit_enabled"])
        self.assertFalse(status["speculation_enabled"])
        self.assertIn("remain failed", status["execution_warning"])
        self.assertEqual(status["context"], 16384)
        self.assertEqual(flags["fit"], "off")
        self.assertEqual(flags["gpu_layers"], "all")
        self.assertIsNone(flags["fit_target"])
        self.assertTrue(flags["cpu_moe"])
        self.assertTrue(flags["no_host"])
        self.assertTrue(flags["no_repack"])
        self.assertTrue(flags["no_op_offload"])
        self.assertEqual(flags["expert_cache_mib"], "10240")
        self.assertEqual(flags["hybrid_plan"], str(self.plan))
        self.assertEqual(flags["cuda_plugin"], str(self.plugin))
        for field in ("hybrid_adapt", "hybrid_async", "hybrid_direct", "register_host"):
            self.assertEqual(flags[field], "1")
        self.assertIsNone(flags["hybrid_batch"])
        self.assertIsNone(flags["hybrid_repack"])
        self.assertIsNone(flags["hybrid_overlap"])
        self.assertIsNone(flags["hybrid_fuse_gate_up"])
        self.assertIsNone(flags["draft_model"])
        process = self.runtime.process
        self.runtime.stop()
        self.assertIsNotNone(process.poll())
        self.assertFalse(self.runtime.status()["hybrid_verified"])
        self.assertTrue(self.load()["hybrid_verified"])

    def precheck(self, model=None):
        code, _, body = self.request("POST", "/api/v1/model/precheck",
                                    {"model": str(model or self.model)})
        self.assertEqual(code, 200, body)
        return json.loads(body)

    def test_precheck_inspects_without_a_profile_and_rejects_invalid_requests(self):
        write_metadata(self.model, HYBRID_METADATA)
        with mock.patch.object(app.subprocess, "Popen") as launch:
            result = self.precheck()
            launch.assert_not_called()
        self.assertFalse(result["hybrid_candidate"])
        self.assertEqual(result["required_hybrid"]["reason_code"], "no_profile")
        self.assertEqual(result["automatic"]["execution_profile"], "standard")
        self.assertEqual(result["layout"]["expert_count"], 128)
        self.assertIn("unverified", " ".join(result["limitations"]))
        self.assertIsNone(self.runtime.process)
        self.assertEqual(self.runtime.state, "stopped")
        for payload in ({}, {"model": False}, {"model": str(self.root / "absent.gguf")},
                        {"model": str(self.model), "execution_mode": "hybrid"}):
            code, _, body = self.request("POST", "/api/v1/model/precheck", payload)
            self.assertEqual(code, 400, body)
        self.assertEqual(self.request("POST", "/api/v1/model/precheck", {"model": str(self.model)},
                                     {"Origin": "https://untrusted.invalid"})[0], 403)
        for raw in (b"not GGUF", b"GGUF", b"GGUF" + struct.pack("<IQQ", 3, 0, 100001)):
            self.model.write_bytes(raw)
            code, _, body = self.request("POST", "/api/v1/model/precheck", {"model": str(self.model)})
            self.assertEqual(code, 400, body)
            self.assertIn(b"error", body)

    def test_precheck_trained_selection_matches_loading_and_does_not_disturb_a_live_model(self):
        self.hybrid_runtime(execution_mode="auto")
        result = self.precheck()
        self.assertTrue(result["hybrid_candidate"])
        self.assertEqual(result["required_hybrid"]["initialization"], "trained")
        status = self.load()
        for key in ("execution_profile", "context"):
            self.assertEqual(status[key], result["automatic"][key])
        self.assertEqual(status["expert_cache_mib"], result["automatic"]["cache_mib"])
        before = {key: status[key] for key in ("model", "state", "selection_reason", "hybrid_verified",
                                               "hybrid_plan_sha256", "logs", "execution_profile")}
        process = self.runtime.process
        dense = self.root / "dense.gguf"
        write_metadata(dense, {"general.architecture": "gemma3"})
        with mock.patch.object(app.subprocess, "Popen") as launch:
            result = self.precheck(dense)
            launch.assert_not_called()
        self.assertFalse(result["hybrid_candidate"])
        self.assertIs(self.runtime.process, process)
        after = self.runtime.status()
        self.assertEqual(before, {key: after[key] for key in before})

    def test_precheck_explains_missing_descriptors_and_bound_plan_mismatches(self):
        self.hybrid_runtime(execution_mode="auto")
        for architecture in ("gemma3", "qwen3", "qwen35", "lfm2", "unknown"):
            with self.subTest(architecture=architecture):
                write_metadata(self.model, {"general.architecture": architecture})
                result = self.precheck()
                self.assertFalse(result["hybrid_candidate"])
                self.assertIsNone(result["layout"])
                self.assertEqual(result["required_hybrid"]["reason_code"], "no_descriptor")
                self.assertIn("may be dense or", result["required_hybrid"]["reason"])
        write_metadata(self.model, {**HYBRID_METADATA, "gpt-oss.expert_used_count": 129})
        self.assertEqual(self.precheck()["required_hybrid"]["reason_code"], "no_descriptor")
        write_metadata(self.model, {**HYBRID_METADATA, "general.file_type": 7})
        result = self.precheck()
        self.assertIsNotNone(result["layout"])
        self.assertEqual(result["required_hybrid"]["reason_code"], "plan_mismatch")
        self.assertIn("trained-plan layout", result["required_hybrid"]["reason"])
        code, _, body = self.request("POST", "/api/v1/model/load",
                                    {"model": str(self.model), "execution_mode": "hybrid"})
        self.assertEqual(code, 400, body)
        self.assertIn(b"trained-plan layout", body)

    def test_precheck_generic_candidate_has_no_architecture_allowlist_and_preserves_native_preference(self):
        self.hybrid_runtime(execution_mode="auto", hybrid_generic=True, hybrid_cache_mib=64,
                            hybrid_context=2048, hybrid_generic_cache_mib=32, hybrid_generic_transfer="staged")
        metadata = {"general.architecture": "future-moe", "future-moe.block_count": 3,
                    "future-moe.embedding_length": 96, "future-moe.feed_forward_length": [64, 96, 128],
                    "future-moe.expert_count": 7, "future-moe.expert_used_count": [1, 2, 3],
                    "future-moe.context_length": 1024}
        write_metadata(self.model, metadata)
        result = self.precheck()
        self.assertTrue(result["hybrid_candidate"])
        self.assertEqual(result["automatic"]["reason_code"], "native_budget")
        self.assertEqual(result["automatic"]["execution_profile"], "standard")
        self.assertEqual(result["required_hybrid"]["context"], 1024)
        self.assertEqual(result["required_hybrid"]["cache_mib"], 32)
        self.assertEqual(result["required_hybrid"]["initialization"], "causal")
        self.assertEqual(result["required_hybrid"]["transfer"], "staged")
        status = self.load()
        self.assertEqual(status["selection_reason"], result["automatic"]["reason"])
        self.runtime.stop()
        for size in (64 * 1024 * 1024, 64 * 1024 * 1024 + 1):
            with self.model.open("r+b") as source:
                source.truncate(size)
            result = self.precheck()
            self.assertEqual(result["automatic"]["reason_code"], "native_budget" if size == 64*1024*1024 else "hybrid")
        status = self.load()
        self.assertEqual(status["selection_reason"], result["automatic"]["reason"])
        self.assertEqual(status["expert_cache_mib"], result["automatic"]["cache_mib"])
        self.runtime.stop()
        write_metadata(self.model, {**metadata, "split.count": 2})
        self.assertEqual(self.precheck()["automatic"]["reason_code"], "hybrid")
        self.assertIn("all split shards", " ".join(self.precheck()["limitations"]))
        write_metadata(self.model, {**metadata, "future-moe.context_length": 1})
        code, _, body = self.request("POST", "/api/v1/model/precheck", {"model": str(self.model)})
        self.assertEqual(code, 400, body)
        self.assertIn(b"context limit", body)

    def test_hybrid_cannot_report_ready_without_verified_execution(self):
        self.hybrid_runtime(load_timeout=0.4)
        for name in ("hybrid-no-markers", "hybrid-wrong-context", "hybrid-partial-coverage"):
            model = self.root / f"{name}.gguf"
            write_metadata(model, HYBRID_METADATA)
            code, _, _ = self.request("POST", "/api/v1/model/load", {"model": str(model)})
            self.assertEqual(code, 202)
            status = self.wait_state("error")
            self.assertIn("Backend context" if name == "hybrid-wrong-context" else "hybrid readiness timed out",
                          status["error"])
            self.assertFalse(status["hybrid_verified"])
            self.runtime.process.wait(timeout=5)
            self.assertIsNotNone(self.runtime.process.poll())

    def test_decode_pipeline_flags_activation_and_standard_rollback(self):
        self.hybrid_runtime(hybrid_decode_pipeline=True)
        status = self.load()
        flags = json.loads(status["logs"][0])
        self.assertTrue(status["hybrid_decode_pipeline"])
        self.assertTrue(status["hybrid_overlap_verified"])
        self.assertTrue(status["hybrid_gate_up_verified"])
        self.assertTrue(status["hybrid_verified"])
        for field in ("hybrid_stage_input", "hybrid_overlap", "hybrid_fuse_gate_up"):
            self.assertEqual(flags[field], "1")
        self.assertEqual(flags["hybrid_boundary"], "gpu")
        self.runtime.stop()
        status = self.load(execution_mode="standard")
        flags = json.loads(status["logs"][0])
        for field in ("hybrid_stage_input", "hybrid_overlap", "hybrid_boundary", "hybrid_fuse_gate_up"):
            self.assertIsNone(flags[field])
        for field in ("hybrid_decode_pipeline", "hybrid_overlap_verified", "hybrid_gate_up_verified"):
            self.assertFalse(status[field])

    def test_decode_pipeline_requires_backend_acknowledgement(self):
        self.hybrid_runtime(hybrid_decode_pipeline=True, load_timeout=0.4)
        model = self.root / "hybrid-no-pipeline.gguf"
        write_metadata(model, HYBRID_METADATA)
        self.request("POST", "/api/v1/model/load", {"model": str(model)})
        status = self.wait_state("error")
        self.assertIn("hybrid readiness timed out", status["error"])
        self.assertFalse(status["hybrid_verified"])

    def test_async_ids_require_verified_backend_and_do_not_leak_into_standard(self):
        self.hybrid_runtime(hybrid_decode_pipeline=True, hybrid_async_ids=True)
        status = self.load()
        self.assertTrue(status["hybrid_async_ids"])
        self.assertTrue(status["hybrid_async_ids_verified"])
        self.assertEqual(json.loads(status["logs"][0])["hybrid_async_ids"], "1")
        self.runtime.stop()
        missing = self.root / "hybrid-no-async-ids.gguf"
        write_metadata(missing, HYBRID_METADATA)
        with mock.patch.object(self.runtime, "load_timeout", 0.4):
            self.request("POST", "/api/v1/model/load", {"model": str(missing)})
            status = self.wait_state("error")
        self.assertFalse(status["hybrid_verified"])
        self.assertIn("async-ids", status["error"])
        self.runtime.stop()
        status = self.load(execution_mode="standard")
        self.assertFalse(status["hybrid_async_ids"])
        self.assertFalse(status["hybrid_async_ids_verified"])
        self.assertIsNone(json.loads(status["logs"][0])["hybrid_async_ids"])

    def test_hosted_moe_requires_markers_and_sets_validated_policy(self):
        self.hybrid_runtime(hybrid_decode_pipeline=True, hybrid_async_ids=True, hybrid_hosted=True)
        status = self.load()
        self.assertTrue(status["hybrid_verified"])
        self.assertTrue(status["hybrid_hosted"])
        self.assertTrue(status["hybrid_hosted_verified"])
        launch = json.loads(status["logs"][0])
        self.assertEqual(launch["hybrid_hosted"], "1")
        self.assertEqual(launch["host_mailbox"], "1")
        self.assertEqual(launch["hybrid_policy"], "lfu")
        self.assertEqual(launch["hybrid_lfu"], ["1", "6"])
        self.assertIsNone(launch["hybrid_stage_input"])
        self.assertIsNone(launch["hybrid_boundary"])
        self.assertEqual(launch["hybrid_async_ids"], "1")
        self.runtime.stop()
        missing = self.root / "hybrid-hosted-missing.gguf"
        write_metadata(missing, HYBRID_METADATA)
        with mock.patch.object(self.runtime, "load_timeout", 0.4):
            self.request("POST", "/api/v1/model/load", {"model": str(missing)})
            status = self.wait_state("error")
        self.assertFalse(status["hybrid_verified"])
        self.assertIn("hosted", status["error"])
        self.runtime.stop()
        status = self.load(execution_mode="standard")
        self.assertFalse(status["hybrid_hosted"])
        launch = json.loads(status["logs"][0])
        self.assertIsNone(launch["hybrid_hosted"])
        self.assertIsNone(launch["host_mailbox"])

    def test_hosted_moe_rejects_staged_transfer_and_mtp(self):
        self.runtime.stop(closing=True)
        self.hybrid_runtime(hybrid_hosted=True, hybrid_transfer="staged")
        code, _, body = self.request("POST", "/api/v1/model/load", {"model": str(self.model)})
        self.assertEqual(code, 400, body)
        self.assertIn(b"direct registered transfers", body)
        self.assertIsNone(self.runtime.process)
        self.runtime.stop(closing=True)
        self.hybrid_runtime(enable_speculation=True, hybrid_generic=True, execution_mode="auto", hybrid_hosted=True,
                            hybrid_generic_transfer="direct")
        draft = self.mtp_models()
        code, _, body = self.request("POST", "/api/v1/model/load", {
            "model": str(self.model), "execution_mode": "hybrid", "draft_type": "mtp",
            "draft_model": str(draft), "draft_tokens": 4, "expert_cache_mib": 64,
        })
        self.assertEqual(code, 400, body)
        self.assertIn(b"not been validated with MTP", body)
        self.assertIsNone(self.runtime.process)

    def test_decode_pipeline_does_not_claim_overlap_from_configuration_alone(self):
        self.hybrid_runtime(hybrid_decode_pipeline=True)
        model = self.root / "hybrid-pipeline-pending.gguf"
        write_metadata(model, HYBRID_METADATA)
        status = self.load(model=str(model))
        self.assertTrue(status["hybrid_verified"])
        self.assertTrue(status["hybrid_decode_pipeline"])
        self.assertFalse(status["hybrid_overlap_verified"])
        self.assertFalse(status["hybrid_gate_up_verified"])

    def test_hybrid_rejects_conflicting_settings_before_spawning(self):
        self.hybrid_runtime()
        for extra in ({"placement": "cpu"}, {"context": 8192}, {"load_mode": "none"},
                      {"expert_cache_mib": 0}, {"host_op_offload": True},
                      {"fit_target_mib": 512}, {"op_offload_min_batch": 128},
                      {"draft_model": str(self.model)}):
            with self.subTest(extra=extra):
                code, _, body = self.request("POST", "/api/v1/model/load", {"model": str(self.model), **extra})
                self.assertEqual(code, 400, body)
                self.assertIsNone(self.runtime.process)

    def test_hybrid_large_context_requires_explicit_profile_budget(self):
        self.hybrid_runtime(hybrid_context=262144)
        code, _, body = self.request("POST", "/api/v1/model/load",
                                    {"model": str(self.model), "context": 65536})
        self.assertEqual(code, 400, body)
        self.assertIn(b"262144 context", body)
        self.assertIsNone(self.runtime.process)
        status = self.load()
        self.assertEqual(status["requested_context"], 262144)
        self.assertEqual(status["context"], 262144)
        self.assertTrue(status["hybrid_verified"])

    def test_hybrid_context_budget_validation(self):
        for context in (True, 511, 262145, 262144.0, "262144"):
            with self.subTest(context=context), self.assertRaises(ValueError):
                app.Runtime(str(self.backend), hybrid_context=context)

    def test_monitor_cursor_validation_and_bounded_endpoint(self):
        for query in ("after=-1", "after=1.5", "after=abc", "after=", "after=1&after=2",
                      "after=99999999999999999", "unknown=0"):
            with self.subTest(query=query):
                self.assertEqual(self.request("GET", "/api/v1/monitor?" + query)[0], 400)
        self.runtime.monitor.stop_event.set()
        self.assertEqual(self.request("GET", "/api/v1/monitor?after=0")[0], 200)

    def test_proxy_observes_actual_timings_without_retaining_content(self):
        self.load()
        code, _, _ = self.request("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "reasoning"}], "stream": True,
            "stream_options": {"include_usage": True}})
        self.assertEqual(code, 200)
        metrics = self.runtime._monitor_backend()
        self.assertEqual(metrics["last_tokens_per_second"], 123.5)
        self.assertEqual(metrics["tokens_per_second"], 123.5)
        self.assertEqual(metrics["speed_source"], "backend_final_timings")
        self.assertEqual(self.runtime._monitor_backend()["tokens_per_second"], 0)
        self.assertEqual(metrics["output_tokens"], 730)
        self.assertFalse(metrics["generation_active"])
        self.assertEqual(metrics["context_source"], "backend_slot")
        self.assertNotIn("Working on the answer", json.dumps(metrics))
        self.assertNotIn("not executable", json.dumps(metrics))
        self.runtime.stop()
        self.assertIsNone(self.runtime._monitor_backend()["last_tokens_per_second"])

    def test_automatic_profile_selects_supported_layout_and_standard_for_other_models(self):
                patched = self.root / "patched-server.py"
                shutil.copyfile(self.backend, patched)
                patched.chmod(0o755)
                self.hybrid_runtime(execution_mode="auto", hybrid_executable=str(patched))
                status = self.load()
                self.assertEqual(status["execution_mode"], "auto")
                self.assertEqual(status["backend"], str(patched))
                self.assertEqual(status["execution_profile"], "experimental-hybrid-direct")
                self.assertTrue(status["hybrid_verified"])
                self.runtime.stop()
                write_metadata(self.model, {"general.architecture": "gemma3", "general.file_type": 15})
                status = self.load()
                self.assertEqual(status["execution_profile"], "standard")
                self.assertEqual(status["backend"], str(self.backend))
                self.assertFalse(status["hybrid_verified"])
                self.assertIn("not supported", status["selection_reason"])
                self.assertIsNone(status["execution_warning"])
                flags = json.loads(status["logs"][0])
                self.assertIsNone(flags["hybrid_plan"])
                self.assertIsNone(flags["register_host"])
                self.runtime.stop()
                code, _, body = self.request("POST", "/api/v1/model/load",
                                            {"model": str(self.model), "execution_mode": "hybrid"})
                self.assertEqual(code, 400, body)
                self.assertIsNone(self.runtime.process)
                write_metadata(self.model, HYBRID_METADATA)
                self.assertTrue(self.load()["hybrid_verified"])

    def test_standard_override_uses_original_backend_even_for_a_supported_model(self):
                self.hybrid_runtime(execution_mode="auto")
                status = self.load(execution_mode="standard")
                self.assertEqual(status["execution_profile"], "standard")
                self.assertFalse(status["hybrid_verified"])
                self.assertIsNone(json.loads(status["logs"][0])["hybrid_plan"])
                self.runtime.stop()
                self.assertEqual(self.load()["execution_profile"], "experimental-hybrid-direct")

    def test_a_different_quantization_or_layout_does_not_reuse_a_bound_training_plan(self):
                self.hybrid_runtime(execution_mode="auto")
                for field, value in (("general.file_type", 7), ("gpt-oss.block_count", 24),
                                     ("gpt-oss.expert_used_count", 8), ("gpt-oss.feed_forward_length", 5760)):
                    write_metadata(self.model, {**HYBRID_METADATA, field: value})
                    status = self.load()
                    self.assertEqual(status["execution_profile"], "standard")
                    self.runtime.stop()
                self.assertFalse(app.supports_hybrid({**HYBRID_METADATA, "gpt-oss.expert_used_count": True}))

    def test_bad_metadata_is_not_a_silent_standard_fallback(self):
                self.hybrid_runtime(execution_mode="auto")
                for data in (b"GGUF", b"GGUF" + struct.pack("<IQQ", 3, 0, 100001),
                             b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 65537)):
                    self.model.write_bytes(data)
                    code, _, body = self.request("POST", "/api/v1/model/load", {"model": str(self.model)})
                    self.assertEqual(code, 400, body)
                    self.assertIn(b"Cannot select", body)
                    self.assertIsNone(self.runtime.process)

    def test_metadata_reader_skips_vocabulary_arrays_and_bounds_allocations(self):
        def string(value):
                    raw = value.encode()
                    return struct.pack("<Q", len(raw)) + raw
        architecture = string("general.architecture") + struct.pack("<I", 8) + string("qwen3")
        vocabulary = (string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) +
                              string("hello") + string("world"))
        self.model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + architecture + vocabulary)
        self.assertEqual(app.model_metadata(self.model), {"general.architecture": "qwen3"})
        for suffix in (struct.pack("<I", 99), struct.pack("<IIQ", 9, 4, 1000001),
                               struct.pack("<IIQ", 9, 4, 999999), struct.pack("<I", 8) + struct.pack("<Q", 2**40)):
                    self.model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) +
                                           architecture + string("unused.key") + suffix)
                    with self.assertRaises(ValueError):
                        app.model_metadata(self.model)
        self.model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + architecture * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
                    app.model_metadata(self.model)

    def test_invalid_installed_profile_fails_before_binding_or_loading(self):
        path = self.root / "invalid-profile.json"
        path.write_text('{"schema_version": 99}')
        result = subprocess.run([sys.executable, str(APP_PATH), "--port", "0", "--no-browser",
                                         "--hybrid-profile", str(path)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Invalid pinned hybrid installation profile", result.stderr)
        self.assertNotIn("SYNAPSE: http", result.stdout)

    def test_persistent_profile_requires_pinned_identity_and_absolute_paths(self):
                self.hybrid_runtime()
                path = self.root / "profile.json"
                profile = {"schema_version": 1, "backend_commit": app.PINNED_LLAMA_COMMIT,
                           "executable": str(self.backend), "plan": str(self.plan), "cuda_plugin": str(self.plugin)}
                path.write_text(json.dumps(profile))
                options = app.read_hybrid_profile(path)
                self.assertEqual(options["hybrid_plan"], self.plan)
                self.assertFalse(options["hybrid_decode_pipeline"])
                path.write_text(json.dumps({**profile, "decode_pipeline": True}))
                self.assertTrue(app.read_hybrid_profile(path)["hybrid_decode_pipeline"])
                path.write_text(json.dumps({**profile, "decode_pipeline": True, "async_ids": True}))
                self.assertTrue(app.read_hybrid_profile(path)["hybrid_async_ids"])
                self.assertFalse(app.read_hybrid_profile(path)["hybrid_hosted"])
                path.write_text(json.dumps({**profile, "hosted_moe": True}))
                self.assertTrue(app.read_hybrid_profile(path)["hybrid_hosted"])
                for changes in ({"schema_version": True}, {"backend_commit": "wrong"},
                                {"executable": "relative"}, {"plan": None}, {"unexpected": 1},
                                {"decode_pipeline": "true"}, {"decode_pipeline": 1},
                                {"async_ids": "true"}, {"async_ids": 1}, {"async_ids": True},
                                {"hosted_moe": "true"}, {"hosted_moe": 1}):
                    path.write_text(json.dumps({**profile, **changes}))
                    with self.assertRaises(ValueError):
                        app.read_hybrid_profile(path)

    def test_generic_layout_detection_does_not_require_a_model_name_or_quantization(self):
        for architecture, experts, active, width, ff in (
                        ("llama", 8, 2, 4096, 14336), ("qwen2moe", 60, 4, 2048, 1408),
                        ("qwen3moe", 128, 8, 2048, 768), ("cohere2moe", 64, 4, 1536, 512),
                        ("future-moe", 7, 3, 96, 192)):
                    metadata = {"general.architecture": architecture, "general.file_type": 15,
                                f"{architecture}.block_count": 48, f"{architecture}.embedding_length": width,
                                f"{architecture}.feed_forward_length": ff, f"{architecture}.expert_count": experts,
                                f"{architecture}.expert_used_count": active}
                    with self.subTest(architecture=architecture):
                        self.assertTrue(app.supports_hybrid(metadata))
                        self.assertEqual(app.moe_layout(metadata)["expert_count"], experts)
        metadata = {"general.architecture": "future-moe", "future-moe.block_count": 3,
                            "future-moe.embedding_length": 96, "future-moe.expert_count": 7,
                            "future-moe.expert_used_count": [1, 2, 3], "future-moe.expert_feed_forward_length": [64, 96, 128]}
        self.assertTrue(app.supports_hybrid(metadata))
        for changes in ({"future-moe.expert_used_count": [1, 2]},
                                {"future-moe.expert_used_count": [1, 2, 8]},
                                {"future-moe.expert_used_count": True}):
                    self.assertFalse(app.supports_hybrid({**metadata, **changes}))

    def test_generic_plan_does_not_reuse_a_different_models_trained_experts(self):
        self.hybrid_runtime(hybrid_generic=True, hybrid_context=2048, hybrid_cache_mib=64)
        other = {"general.architecture": "qwen2moe", "general.file_type": 7,
                         "qwen2moe.block_count": 3, "qwen2moe.embedding_length": 96,
                         "qwen2moe.feed_forward_length": 192, "qwen2moe.expert_count": 7, "qwen2moe.expert_used_count": 3,
                         "qwen2moe.context_length": 1024}
        write_metadata(self.model, other)
        status = self.load(execution_mode="hybrid")
        self.assertEqual(status["execution_profile"], "experimental-hybrid-staged")
        self.assertTrue(status["hybrid_verified"])
        self.assertEqual(status["context"], 1024)
        self.assertEqual(status["hybrid_context"], 1024)
        self.assertEqual(status["hybrid_context_budget"], 2048)
        self.assertEqual(status["expert_cache_mib"], 64)
        self.assertIsNone(status["hybrid_plan_sha256"])
        flags = json.loads(status["logs"][0])
        self.assertEqual(flags["hybrid_plan"], "auto")
        self.assertIsNone(flags["hybrid_direct"])
        self.assertIsNone(flags["register_host"])
        self.runtime.stop()
        status = self.load(execution_mode="auto")
        self.assertEqual(status["execution_profile"], "standard")
        self.assertIn("native fitting is preferred", status["selection_reason"])
        self.runtime.stop()
        with self.model.open("r+b") as model:
                    model.truncate(64*1024*1024 + 1)
        status = self.load(execution_mode="auto")
        self.assertEqual(status["execution_profile"], "experimental-hybrid-staged")
        self.assertTrue(status["hybrid_verified"])

    def test_rejected_concurrent_model_load_does_not_poison_active_plan_markers(self):
        self.hybrid_runtime(hybrid_generic=True)
        slow = self.root / "slow.gguf"
        write_metadata(slow, HYBRID_METADATA)
        code, _, _ = self.request("POST", "/api/v1/model/load", {"model": str(slow), "execution_mode": "hybrid"})
        self.assertEqual(code, 202)
        before = self.runtime.status()
        expected_experts = self.runtime.hybrid_experts
        write_metadata(self.model, {"general.architecture": "qwen2moe", "general.file_type": 7,
                                   "qwen2moe.block_count": 3, "qwen2moe.embedding_length": 96,
                                   "qwen2moe.feed_forward_length": 192, "qwen2moe.expert_count": 7,
                                   "qwen2moe.expert_used_count": 3})
        code, _, _ = self.request("POST", "/api/v1/model/load", {"model": str(self.model), "execution_mode": "hybrid"})
        self.assertEqual(code, 409)
        self.assertEqual(self.runtime.hybrid_experts, expected_experts)
        self.assertEqual(self.runtime.status()["hybrid_plan_sha256"], before["hybrid_plan_sha256"])
        self.assertEqual(self.runtime.status()["model"], str(slow))

    def test_manual_causal_budget_does_not_change_automatic_or_bound_plan_budgets(self):
        self.hybrid_runtime(hybrid_generic=True, hybrid_context=2048, hybrid_cache_mib=64,
                            hybrid_generic_cache_mib=32, hybrid_generic_transfer="direct")
        other = {"general.architecture": "future-moe", "future-moe.block_count": 3,
                 "future-moe.embedding_length": 96, "future-moe.feed_forward_length": 192,
                 "future-moe.expert_count": 7, "future-moe.expert_used_count": 3,
                 "future-moe.context_length": 1024, "split.count": 2}
        write_metadata(self.model, other)
        status = self.load(execution_mode="hybrid", expert_cache_mib=24)
        self.assertEqual(status["expert_cache_mib"], 24)
        self.assertEqual(status["hybrid_generic_cache_mib"], 32)
        self.assertEqual(status["hybrid_cache_mib"], 64)
        self.assertIn("Explicit causal residency budget: 24 MiB", status["selection_reason"])
        flags = json.loads(status["logs"][0])
        self.assertEqual(flags["hybrid_plan"], "auto")
        self.assertEqual(flags["expert_cache_mib"], "24")
        self.runtime.stop()
        for value in (0, -1, True, 65537):
            code, _, _ = self.request("POST", "/api/v1/model/load",
                                     {"model": str(self.model), "execution_mode": "hybrid", "expert_cache_mib": value})
            self.assertEqual(code, 400)
        code, _, _ = self.request("POST", "/api/v1/model/load",
                                 {"model": str(self.model), "execution_mode": "auto", "expert_cache_mib": 24})
        self.assertEqual(code, 400)
        write_metadata(self.model, HYBRID_METADATA)
        code, _, _ = self.request("POST", "/api/v1/model/load",
                                 {"model": str(self.model), "execution_mode": "hybrid", "expert_cache_mib": 24})
        self.assertEqual(code, 400)
        status = self.load(execution_mode="hybrid")
        self.assertEqual(status["expert_cache_mib"], 64)
        self.assertEqual(json.loads(status["logs"][0])["hybrid_plan"], str(self.plan))

    def test_generic_budget_and_direct_transfer_are_independent_of_bound_plan(self):
        self.hybrid_runtime(hybrid_generic=True, hybrid_context=2048, hybrid_cache_mib=64,
                            hybrid_generic_cache_mib=32, hybrid_generic_transfer="direct")
        other = {"general.architecture": "future-moe", "future-moe.block_count": 3,
                 "future-moe.embedding_length": 96, "future-moe.feed_forward_length": 192,
                 "future-moe.expert_count": 7, "future-moe.expert_used_count": 3,
                 "future-moe.context_length": 1024}
        write_metadata(self.model, other)
        status = self.load(execution_mode="hybrid")
        self.assertEqual(status["expert_cache_mib"], 32)
        self.assertEqual(status["hybrid_cache_mib"], 64)
        self.assertEqual(status["hybrid_generic_cache_mib"], 32)
        self.assertEqual(status["hybrid_generic_transfer"], "direct")
        self.assertEqual(status["execution_profile"], "experimental-hybrid-direct")
        self.assertTrue(status["hybrid_verified"])
        flags = json.loads(status["logs"][0])
        self.assertEqual(flags["hybrid_plan"], "auto")
        self.assertEqual(flags["expert_cache_mib"], "32")
        self.assertEqual(flags["hybrid_direct"], "1")
        self.assertEqual(flags["register_host"], "1")
        self.runtime.stop()
        write_metadata(self.model, HYBRID_METADATA)
        status = self.load(execution_mode="hybrid")
        self.assertEqual(status["expert_cache_mib"], 64)
        self.assertEqual(status["hybrid_plan_sha256"], self.runtime.hybrid_plan_sha256)
        self.assertEqual(json.loads(status["logs"][0])["hybrid_plan"], str(self.plan))

    def test_generic_budget_requests_do_not_poison_live_model(self):
        self.hybrid_runtime(hybrid_generic=True, hybrid_cache_mib=64, hybrid_generic_cache_mib=32)
        other = {"general.architecture": "future-moe", "future-moe.block_count": 3,
                 "future-moe.embedding_length": 96, "future-moe.feed_forward_length": 192,
                 "future-moe.expert_count": 7, "future-moe.expert_used_count": 3, "split.count": 2}
        write_metadata(self.model, other)
        before = self.load(execution_mode="hybrid")
        for mode, expected_code, message in (("auto", 400, "32 MiB residency"),
                                             ("hybrid", 409, "Unload the current model")):
            with self.subTest(mode=mode):
                code, _, body = self.request("POST", "/api/v1/model/load",
                                            {"model": str(self.model), "execution_mode": mode,
                                             "expert_cache_mib": 64})
                self.assertEqual(code, expected_code, body)
                self.assertIn(message, body.decode())
                status = self.runtime.status()
                self.assertEqual(status["hybrid_plan_sha256"], before["hybrid_plan_sha256"])
                self.assertEqual(status["expert_cache_mib"], 32)
                self.assertTrue(status["hybrid_verified"])

    def test_generic_options_are_validated_and_preserve_auto_plan_defaults(self):
        self.hybrid_runtime()
        profile = {"schema_version": 2, "backend_commit": app.PINNED_LLAMA_COMMIT,
                   "executable": str(self.backend), "plan": "auto", "cuda_plugin": str(self.plugin),
                   "context": 2048, "cache_mib": 64, "transfer": "direct"}
        path = self.root / "generic-options.json"
        path.write_text(json.dumps(profile))
        runtime = app.Runtime(str(self.backend), **app.read_hybrid_profile(path))
        self.assertEqual(runtime.hybrid_generic_cache_mib, 64)
        self.assertEqual(runtime.active_hybrid_transfer, "direct")
        for changes in ({"generic_cache_mib": None}, {"generic_cache_mib": True},
                        {"generic_cache_mib": 0}, {"generic_cache_mib": 65537},
                        {"generic_cache_mib": "32"}, {"generic_transfer": None},
                        {"generic_transfer": "unknown"}, {"generic_transfer": True}):
            path.write_text(json.dumps({**profile, **changes}))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                app.read_hybrid_profile(path)
        path.write_text(json.dumps({**profile, "generic_cache_mib": 32, "generic_transfer": "staged"}))
        runtime = app.Runtime(str(self.backend), **app.read_hybrid_profile(path))
        self.assertEqual(runtime.expert_cache_mib, 32)
        self.assertEqual(runtime.active_hybrid_transfer, "staged")
        for changes in ({"hybrid_generic_cache_mib": True}, {"hybrid_generic_cache_mib": 0},
                        {"hybrid_generic_cache_mib": 65537}, {"hybrid_generic_transfer": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                app.Runtime(str(self.backend), **changes)

    def test_generic_installation_profile_accepts_causal_initialization_and_binds_trained_plans(self):
        self.hybrid_runtime()
        path = self.root / "generic-profile.json"
        profile = {"schema_version": 2, "backend_commit": app.PINNED_LLAMA_COMMIT,
                           "executable": str(self.backend), "plan": "auto", "cuda_plugin": str(self.plugin),
                           "context": 2048, "cache_mib": 64, "transfer": "staged"}
        path.write_text(json.dumps(profile))
        options = app.read_hybrid_profile(path)
        self.assertEqual(options["hybrid_plan"], "auto")
        self.assertTrue(options["hybrid_generic"])
        runtime = app.Runtime(str(self.backend), execution_mode="auto", **options)
        self.assertEqual(runtime.hybrid_cache_mib, 64)
        self.assertIsNone(runtime.hybrid_plan_sha256)
        path.write_text(json.dumps({**profile, "plan": str(self.plan)}))
        with self.assertRaisesRegex(ValueError, "requires model_layout"):
                    app.read_hybrid_profile(path)
        path.write_text(json.dumps({**profile, "plan": str(self.plan), "model_layout": app.moe_layout(HYBRID_METADATA)}))
        self.assertEqual(app.read_hybrid_profile(path)["hybrid_layout"], app.moe_layout(HYBRID_METADATA))

    def test_native_backend_context_adjustment_is_visible_and_reports_actual_size(self):
        model = self.root / "context-cap.gguf"
        model.write_bytes(b"GGUF")
        status = self.load(model=model, context=4096)
        self.assertEqual(status["requested_context"], 4096)
        self.assertEqual(status["context"], 2048)
        self.assertIn("adjusted", status["context_warning"])
        self.assertEqual(status["state"], "ready")

    def test_hybrid_validates_launch_paths_and_plan(self):
        self.hybrid_runtime()
        for extra in ({"hybrid_plan": self.plan}, {"hybrid_cuda_plugin": self.plugin},
                      {"hybrid_plan": self.plan, "hybrid_cuda_plugin": self.plugin,
                       "backend_kind": "ik_llama.cpp"}):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    app.Runtime(str(self.backend), **extra)
        for content in ("", "0 7\n0 7\n", "36 0\n", "0 128\n", "not a plan\n", "#"*(1024*1024 + 1)):
            self.plan.write_text(content)
            with self.assertRaises(ValueError):
                app.Runtime(str(self.backend), hybrid_plan=self.plan, hybrid_cuda_plugin=self.plugin,
                            hybrid_layout=app.moe_layout(HYBRID_METADATA))
        code, _, body = self.request("POST", "/api/v1/model/load", {"model": str(self.model)})
        self.assertEqual(code, 400, body)
        self.assertIsNone(self.runtime.process)

    def test_host_execution_controls_are_explicit(self):
        status = self.load(host_op_offload=False, cpu_poll=0, op_offload_min_batch=128)
        self.assertFalse(status["host_op_offload"])
        self.assertEqual(status["cpu_poll"], 0)
        flags = json.loads(status["logs"][0])
        self.assertTrue(flags["no_op_offload"])
        self.assertEqual(flags["poll"], "0")
        self.assertEqual(status["op_offload_min_batch"], 128)
        self.assertEqual(flags["op_offload_min_batch"], "128")

    def test_host_execution_defaults_preserve_backend_behavior(self):
        status = self.load()
        flags = json.loads(status["logs"][0])
        self.assertFalse(flags["no_op_offload"])
        self.assertEqual(flags["poll"], "50")
        self.assertEqual(flags["op_offload_min_batch"], "32")

    def test_offload_threshold_cannot_be_overridden_by_shell(self):
        with mock.patch.dict(os.environ, {"GGML_OP_OFFLOAD_MIN_BATCH": "1"}):
            status = self.load()
        self.assertEqual(json.loads(status["logs"][0])["op_offload_min_batch"], "32")

    def test_invalid_host_execution_controls_are_rejected(self):
        for extra in ({"host_op_offload": 0}, {"host_op_offload": "false"},
                      {"cpu_poll": True}, {"cpu_poll": -1}, {"cpu_poll": 101},
                      {"op_offload_min_batch": True}, {"op_offload_min_batch": 0},
                      {"op_offload_min_batch": 8193}):
            code, _, _ = self.request("POST", "/api/v1/model/load", {"model": str(self.model), **extra})
            self.assertEqual(code, 400)
        self.runtime.backend_kind = "ik_llama.cpp"
        for extra in ({"host_op_offload": False}, {"cpu_poll": 0}, {"op_offload_min_batch": 128}):
            code, _, _ = self.request("POST", "/api/v1/model/load", {"model": str(self.model), **extra})
            self.assertEqual(code, 400)
        self.assertIsNone(self.runtime.process)

    def test_explicit_execution_settings_and_cache_flags(self):
        threads = min(2, os.cpu_count() or 1)
        status = self.load(cpu_threads=threads, batch_threads=1, load_mode="none",
                           expert_cache_mib=8192)
        self.assertEqual(status["cpu_threads"], threads)
        self.assertEqual(status["batch_threads"], 1)
        self.assertEqual(status["load_mode"], "none")
        self.assertEqual(status["expert_cache_mib"], 8192)
        flags = json.loads(status["logs"][0])
        self.assertEqual(flags["threads"], str(threads))
        self.assertEqual(flags["batch_threads"], "1")
        self.assertEqual(flags["load_mode"], "none")
        self.assertEqual(flags["expert_cache_mib"], "8192")
        self.assertEqual(flags["gpu_layers"], "all")
        self.assertTrue(flags["cpu_moe"])

    def test_invalid_execution_settings(self):
        for extra in ({"cpu_threads": True}, {"cpu_threads": 0},
                      {"cpu_threads": (os.cpu_count() or 1) + 1},
                      {"batch_threads": 0}, {"batch_threads": "4"},
                      {"load_mode": "unrestricted"}, {"expert_cache_mib": -1},
                      {"expert_cache_mib": True}, {"expert_cache_mib": 65537},
                      {"expert_cache_mib": 1024, "placement": "cpu"}):
            code, _, body = self.request("POST", "/api/v1/model/load",
                                        {"model": str(self.model), **extra})
            self.assertEqual(code, 400, body)
        self.assertIsNone(self.runtime.process)

    def test_default_threads_count_physical_cores_within_affinity(self):
        def topology(path, *args, **kwargs):
            cpu = int(path.parent.parent.name.removeprefix("cpu"))
            return "0" if path.name == "physical_package_id" else str(cpu % 2)
        with mock.patch.object(app, "available_cpus", return_value={0, 1, 2, 3}), \
                mock.patch.object(app.sys, "platform", "linux"):
            with mock.patch.object(Path, "read_text", autospec=True, side_effect=topology):
                self.assertEqual(app.default_cpu_threads(), 2)

    def test_portable_thread_default_does_not_read_linux_topology(self):
        with mock.patch.object(app, "available_cpus", return_value={0, 1, 2, 3}), \
                mock.patch.object(app.sys, "platform", "darwin"), \
                mock.patch.object(Path, "read_text", side_effect=AssertionError("Linux topology read")):
            self.assertEqual(app.default_cpu_threads(), 4)

    def test_explicit_experimental_backend_adapter(self):
        self.runtime.backend_kind = "ik_llama.cpp"
        status = self.load(load_mode="none")
        self.assertEqual(status["backend_kind"], "ik_llama.cpp")
        flags = json.loads(status["logs"][0])
        self.assertEqual(flags["gpu_layers"], "999")
        self.assertEqual(flags["fit"], "enabled")
        self.assertEqual(flags["fit_margin"], "1024")
        self.assertTrue(flags["no_mmap"])
        self.assertIsNone(flags["fit_target"])

    def test_experimental_backend_rejects_unverified_modes(self):
        self.runtime.backend_kind = "ik_llama.cpp"
        for extra in ({"placement": "cpu"}, {"expert_cache_mib": 1024}):
            code, _, body = self.request("POST", "/api/v1/model/load",
                                        {"model": str(self.model), **extra})
            self.assertEqual(code, 400, body)
            self.assertIn(b"experimental ik_llama.cpp", body)
        self.assertIsNone(self.runtime.process)

    def test_dflash_flags_are_explicit_and_backend_specific(self):
        self.runtime.enable_speculation = True
        draft = self.root / "draft.gguf"
        draft.write_bytes(b"GGUF")
        status = self.load(draft_model=str(draft), draft_tokens=5, fit_target_mib=4096)
        flags = json.loads(status["logs"][0])
        self.assertEqual(status["draft_model"], str(draft))
        self.assertEqual(status["draft_type"], "dflash")
        self.assertTrue(status["speculation_verified"])
        self.assertEqual(status["fit_target_mib"], 4096)
        self.assertEqual(flags["spec_type"], "draft-dflash")
        self.assertEqual(flags["draft_tokens"], "5")
        self.assertEqual(flags["draft_gpu_layers"], "all")
        self.assertEqual(flags["fit_target"], "4096")
        self.assertTrue(flags["jinja"])
        self.assertEqual(flags["flash_attn"], "on")
        self.runtime.stop()
        self.runtime.backend_kind = "ik_llama.cpp"
        status = self.load(draft_model=str(draft), draft_tokens=9, fit_target_mib=4096)
        flags = json.loads(status["logs"][0])
        self.assertEqual(flags["spec_type"], "dflash:n_max=9,p_min=0.0")
        self.assertEqual(flags["draft_gpu_layers"], "999")
        self.assertEqual(flags["fit_margin"], "4096")

    def test_invalid_dflash_configuration(self):
        self.runtime.enable_speculation = True
        draft = self.root / "draft.gguf"
        draft.write_bytes(b"GGUF")
        for extra in ({"draft_model": "/missing.gguf"}, {"draft_model": str(self.model)},
                      {"draft_tokens": True}, {"draft_tokens": 0}, {"draft_tokens": 33},
                      {"fit_target_mib": True}, {"fit_target_mib": 127},
                      {"draft_model": str(draft), "placement": "cpu"},
                      {"draft_model": str(draft), "expert_cache_mib": 1024}):
            code, _, body = self.request("POST", "/api/v1/model/load",
                                        {"model": str(self.model), **extra})
            self.assertEqual(code, 400, body)
        self.assertIsNone(self.runtime.process)

    def test_dflash_is_disabled_by_default(self):
        draft = self.root / "draft.gguf"
        draft.write_bytes(b"GGUF")
        code, _, body = self.request("POST", "/api/v1/model/load",
                                    {"model": str(self.model), "draft_model": str(draft)})
        self.assertEqual(code, 400)
        self.assertIn(b"--experimental-speculation", body)
        self.assertFalse(self.runtime.status()["speculation_enabled"])
        self.assertIsNone(self.runtime.process)

    def mtp_models(self):
        write_metadata(self.model, MTP_METADATA)
        draft = self.root / "native-mtp.gguf"
        write_metadata(draft, {**MTP_METADATA, "qwen4exp.block_count": 3,
                               "qwen4exp.nextn_predict_layers": 1})
        return draft

    def test_mtp_is_explicit_and_verified_in_standard_mode(self):
        self.runtime.enable_speculation = True
        draft = self.mtp_models()
        status = self.load(draft_model=str(draft), draft_type="mtp")
        flags = json.loads(status["logs"][0])
        self.assertEqual(flags["spec_type"], "draft-mtp")
        self.assertEqual(flags["draft_tokens"], "4")
        self.assertEqual(flags["draft_gpu_layers"], "all")
        self.assertFalse(flags["jinja"])
        self.assertIsNone(flags["flash_attn"])
        self.assertEqual(status["draft_type"], "mtp")
        self.assertTrue(status["speculation_verified"])
        self.runtime.stop()
        status = self.load()
        self.assertEqual(status["draft_type"], "none")
        self.assertFalse(status["speculation_verified"])
        self.assertIsNone(json.loads(status["logs"][0])["draft_model"])

    def test_hybrid_mtp_uses_explicit_native_optin_and_bounded_target_batch(self):
        self.runtime.stop(closing=True)
        self.hybrid_runtime(enable_speculation=True, hybrid_generic=True, execution_mode="auto")
        draft = self.mtp_models()
        with mock.patch.dict(os.environ, {"VRAMBRIDGE_HYBRID_MTP": "0", "VRAMBRIDGE_HYBRID_BATCH": "32"}):
            status = self.load(execution_mode="hybrid", draft_type="mtp", draft_model=str(draft),
                               draft_tokens=4, expert_cache_mib=64)
        flags = json.loads(status["logs"][0])
        self.assertEqual(flags["spec_type"], "draft-mtp")
        self.assertEqual(flags["hybrid_mtp"], "1")
        self.assertEqual(flags["hybrid_batch"], "5")
        self.assertEqual(flags["hybrid_plan"], "auto")
        self.assertEqual(status["draft_type"], "mtp")
        self.assertTrue(status["hybrid_verified"])
        self.assertTrue(status["speculation_verified"])
        self.runtime.stop()
        code, _, body = self.request("POST", "/api/v1/model/load", {
            "model": str(self.model), "execution_mode": "hybrid", "draft_type": "mtp",
            "draft_model": str(draft), "draft_tokens": 32, "expert_cache_mib": 64,
        })
        self.assertEqual(code, 400, body)
        self.assertIn(b"31 draft tokens", body)
        self.assertIsNone(self.runtime.process)

    def test_configuring_speculation_does_not_enable_a_draft_and_hybrid_dflash_stays_blocked(self):
        self.runtime.stop(closing=True)
        self.hybrid_runtime(enable_speculation=True)
        status = self.load()
        self.assertTrue(status["hybrid_verified"])
        self.assertEqual(status["draft_type"], "none")
        self.assertIsNone(status["draft_model"])
        self.assertIsNone(json.loads(status["logs"][0])["hybrid_mtp"])
        self.runtime.stop()
        draft = self.root / "dflash.gguf"
        draft.write_bytes(b"GGUF")
        code, _, body = self.request("POST", "/api/v1/model/load", {
            "model": str(self.model), "draft_type": "dflash", "draft_model": str(draft),
        })
        self.assertEqual(code, 400, body)
        self.assertIn(b"Standard mode for DFlash", body)
        self.assertIsNone(self.runtime.process)

    def test_invalid_mtp_configuration_fails_before_starting_backend(self):
        self.runtime.enable_speculation = True
        draft = self.mtp_models()
        for extra in ({"draft_type": "unknown"}, {"draft_type": ["mtp"]},
                      {"draft_type": "mtp"}, {"draft_type": "none", "draft_model": str(draft)},
                      {"draft_type": "mtp", "draft_model": str(self.model)},
                      {"draft_type": "mtp", "draft_model": str(draft), "placement": "cpu"}):
            code, _, body = self.request("POST", "/api/v1/model/load", {"model": str(self.model), **extra})
            self.assertEqual(code, 400, body)
            self.assertIsNone(self.runtime.process)
        for changes in ({"general.architecture": "qwen4exp-mtp"},
                        {"qwen4exp.embedding_length": 128},
                        {"qwen4exp.nextn_predict_layers": 0},
                        {"qwen4exp.vocab_size": 32}):
            write_metadata(draft, {**MTP_METADATA, "qwen4exp.nextn_predict_layers": 1, **changes})
            code, _, body = self.request("POST", "/api/v1/model/load", {
                "model": str(self.model), "draft_type": "mtp", "draft_model": str(draft),
            })
            self.assertEqual(code, 400, body)
            self.assertIn(b"Cannot use draft model", body)
            self.assertIsNone(self.runtime.process)
        self.runtime.backend_kind = "ik_llama.cpp"
        code, _, body = self.request("POST", "/api/v1/model/load", {
            "model": str(self.model), "draft_type": "mtp", "draft_model": str(draft),
        })
        self.assertEqual(code, 400, body)
        self.assertIn(b"official llama.cpp", body)

    def test_mtp_target_only_fallback_cannot_be_marked_ready(self):
        self.runtime.enable_speculation = True
        draft = self.mtp_models()
        model = self.root / "mtp-fallback.gguf"
        write_metadata(model, MTP_METADATA)
        with mock.patch.object(app, "SPECULATION_VERIFICATION_TIMEOUT", 0.1):
            code, _, body = self.request("POST", "/api/v1/model/load", {
                "model": str(model), "draft_type": "mtp", "draft_model": str(draft),
            })
            self.assertEqual(code, 202, body)
            with self.assertLogs("synapse", level="ERROR"):
                status = self.wait_state("error")
        self.assertFalse(status["speculation_verified"])
        self.assertIn("Target-only fallback is not accepted", status["error"])
        self.assertIsNotNone(self.runtime.process.poll())

    def test_auto_tune_cancellation_restores_mtp_method_and_verified_draft(self):
        self.runtime.enable_speculation = True
        draft = self.mtp_models()
        write_metadata(self.model, {**MTP_METADATA, "qwen4exp.context_length": 8192})
        self.load(draft_type="mtp", draft_model=str(draft), context=4096)
        apply = self.server.tuner._apply

        def cancel_apply(runtime, settings, budget, *, restoring):
            if not restoring:
                self.server.tuner.cancel()
            return apply(runtime, settings, budget, restoring=restoring)

        with mock.patch.object(self.server.tuner, "_apply", side_effect=cancel_apply):
            self.assertEqual(self.request("POST", "/api/v1/tuning/start",
                                         {"model": str(self.model), "confirm": True})[0], 202)
            result = self.wait_tuning("cancelled")
        self.assertIsNone(result["recommendation"])
        status = self.runtime.status()
        self.assertEqual(status["draft_type"], "mtp")
        self.assertEqual(status["draft_model"], str(draft))
        self.assertEqual(status["context"], 4096)
        self.assertTrue(status["speculation_verified"])

    def test_hybrid_mtp_requires_backend_confirmation_of_target_verification_batch(self):
        self.runtime.stop(closing=True)
        self.hybrid_runtime(enable_speculation=True, hybrid_generic=True)
        draft = self.mtp_models()
        model = self.root / "hybrid-mtp-no-batch.gguf"
        write_metadata(model, MTP_METADATA)
        with mock.patch.object(app, "HYBRID_VERIFICATION_TIMEOUT", 0.1):
            code, _, body = self.request("POST", "/api/v1/model/load", {
                "model": str(model), "draft_type": "mtp", "draft_model": str(draft),
                "expert_cache_mib": 64, "execution_mode": "hybrid",
            })
            self.assertEqual(code, 202, body)
            with self.assertLogs("synapse", level="ERROR"):
                status = self.wait_state("error")
        self.assertTrue(status["speculation_verified"])
        self.assertFalse(status["hybrid_verified"])
        self.assertIn("Unconfirmed: mtp", status["error"])

    def test_process_resource_usage(self):
        self.load()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = self.runtime.status()
            resources = status["resources"]
            if resources and resources["cpu_logical_cores"] is not None:
                break
            time.sleep(0.05)
        self.assertIsNotNone(resources)
        self.assertGreater(resources["rss_bytes"], 0)
        for key in ("anonymous_bytes", "file_backed_bytes"):
            if resources[key] is not None:
                self.assertGreaterEqual(resources[key], 0)
        self.assertGreater(resources["system_available_ram_bytes"], 0)
        self.assertGreaterEqual(resources["cpu_logical_cores"], 0)
        self.assertIsNone(status["resource_error"])
        self.runtime.stop()
        self.assertIsNone(self.runtime.status()["resources"])

    def test_resource_sampling_failure_is_visible(self):
        self.load()
        with mock.patch.object(app.system_probe(), "process", side_effect=PermissionError("test denied")):
            self.runtime._sample_resources(self.runtime.process)
            self.assertIn("test denied", self.runtime.status()["resource_error"])
            self.assertIsNone(self.runtime.status()["resources"])
            self.assertEqual(self.runtime.status()["state"], "ready")

    def test_invalid_load_inputs(self):
        for values in ({}, {"model": 5}, {"model": "/missing.gguf"},
                       {"model": str(self.model), "context": True},
                       {"model": str(self.model), "context": 511},
                       {"model": str(self.model), "context": 262145},
                       {"model": str(self.model), "context": 262144.0},
                       {"model": str(self.model), "context": "262144"},
                       {"model": str(self.model), "placement": "magic"}):
            self.assertEqual(self.request("POST", "/api/v1/model/load", values)[0], 400)
        self.model.write_bytes(b"not-a-model")
        self.assertEqual(self.request("POST", "/api/v1/model/load", {"model": str(self.model)})[0], 400)

    def test_backend_failure_and_recovery(self):
        failed = self.root / "fail.gguf"
        failed.write_bytes(b"GGUF")
        self.assertEqual(self.request("POST", "/api/v1/model/load", {"model": str(failed)})[0], 202)
        status = self.wait_state("error")
        self.assertIn("code 7", status["error"])
        self.load()
        self.assertIsNone(self.runtime.status()["error"])

    def test_loading_is_cancellable(self):
        slow = self.root / "slow.gguf"
        slow.write_bytes(b"GGUF")
        self.assertEqual(self.request("POST", "/api/v1/model/load", {"model": str(slow)})[0], 202)
        process = self.runtime.process
        self.request("POST", "/api/v1/model/unload", {})
        self.assertIsNotNone(process.poll())
        time.sleep(0.3)
        self.assertEqual(self.runtime.status()["state"], "stopped")

    def test_loading_timeout(self):
        slow = self.root / "slow.gguf"
        slow.write_bytes(b"GGUF")
        self.runtime.load_timeout = 0.1
        self.request("POST", "/api/v1/model/load", {"model": str(slow)})
        status = self.wait_state("error")
        self.assertIn("timed out", status["error"])
        self.assertIsNotNone(self.runtime.process.poll())

    def test_backend_crash_after_ready(self):
        self.load()
        self.runtime.process.terminate()
        self.runtime.process.wait(timeout=5)
        self.assertIn("exited", self.wait_state("error")["error"])
        self.assertEqual(self.request("GET", "/v1/models")[0], 503)

    def test_gpu_init_failure_is_not_a_successful_cpu_fallback(self):
        model = self.root / "gpu-fail.gguf"
        model.write_bytes(b"GGUF")
        code, _, _ = self.request("POST", "/api/v1/model/load", {"model": str(model)})
        self.assertEqual(code, 202)
        status = self.wait_state("error")
        self.assertIn("CUDA initialization failed", status["error"])
        self.assertIn("CPU-only", status["error"])
        self.assertEqual(self.request("GET", "/v1/models")[0], 503)
        self.runtime.process.wait(timeout=6)
        self.assertIsNotNone(self.runtime.process.poll())
        self.assertEqual(self.runtime.status()["state"], "error")

    def test_explicit_cpu_mode_does_not_require_cuda(self):
        model = self.root / "gpu-fail.gguf"
        model.write_bytes(b"GGUF")
        status = self.load(model=model, placement="cpu")
        self.assertEqual(status["state"], "ready")
        self.assertIsNone(status["error"])

    def test_chat_and_backend_errors(self):
        self.load()
        payload = {"messages": [{"role": "user", "content": "hello"}], "model": "anything"}
        code, _, body = self.request("POST", "/v1/chat/completions", payload)
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["model"], "local-model")
        payload["messages"][0]["content"] = "backend-error"
        code, _, body = self.request("POST", "/v1/chat/completions", payload)
        self.assertEqual(code, 400)
        self.assertIn(b"Context limit", body)

    def test_stream_and_single_request_admission(self):
        self.load()
        payload = {"messages": [{"role": "user", "content": "hello"}], "stream": True}
        code, headers, body = self.request("POST", "/v1/chat/completions", payload)
        self.assertEqual(code, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        self.assertIn(b"data: [DONE]", body)
        self.assertIn(b"<script>not executable</script>", body)
        self.runtime.chat_lock.acquire()
        try:
            self.assertEqual(self.request("POST", "/v1/chat/completions", payload)[0], 409)
        finally:
            self.runtime.chat_lock.release()

    def test_disconnect_cancels_stream(self):
        self.load()
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = {"messages": [{"role": "user", "content": "wait"}], "stream": True}
        connection.request("POST", "/v1/chat/completions", json.dumps(payload),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertTrue(self.runtime.chat_lock.locked())
        response.close()
        connection.close()
        deadline = time.monotonic() + 2
        while self.runtime.chat_lock.locked() and time.monotonic() < deadline:
            time.sleep(0.03)
        self.assertFalse(self.runtime.chat_lock.locked())

    def test_reasoning_usage_and_timings_are_forwarded(self):
        self.load()
        for prompt in ("reasoning", "reasoning-limit"):
            payload = {
                "messages": [{"role": "user", "content": prompt}], "stream": True,
                "max_tokens": 8192, "stream_options": {"include_usage": True},
                "timings_per_token": True,
            }
            code, _, body = self.request("POST", "/v1/chat/completions", payload)
            self.assertEqual(code, 200)
            events = [json.loads(line[6:]) for line in body.splitlines()
                      if line.startswith(b"data: ") and line != b"data: [DONE]"]
            self.assertEqual(events[0]["choices"][0]["delta"]["reasoning_content"],
                             "Working on the answer.")
            self.assertEqual(events[-1]["usage"]["completion_tokens"], 730)
            self.assertEqual(events[-2]["timings"]["predicted_per_second"], 123.5)
            self.assertEqual(events[-2]["choices"][0]["finish_reason"],
                             "length" if prompt == "reasoning-limit" else "stop")

    def test_host_and_origin_restrictions(self):
        for headers in ({"Host": "attacker.example"},
                        {"Origin": "https://attacker.example"},
                        {"Origin": "null"},
                        {"Sec-Fetch-Site": "cross-site"}):
            self.assertEqual(self.request("GET", "/api/v1/status", headers=headers)[0], 403)
            self.assertEqual(self.request("POST", "/api/v1/model/unload", {}, headers)[0], 403)
        self.assertEqual(self.request("GET", "/api/v1/status",
                                     headers={"Origin": f"http://127.0.0.1:{self.port}"})[0], 200)

    def test_request_validation(self):
        self.assertEqual(self.request("POST", "/api/v1/model/load", [])[0], 400)
        self.assertEqual(self.request("POST", "/api/v1/model/load", {},
                                     {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/v1/model/load", {},
                                     {"Transfer-Encoding": "chunked"})[0], 400)
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"messages": []})[0], 400)
        self.assertEqual(self.request("POST", "/v1/chat/completions",
                                     {"messages": [{}], "stream": "yes"})[0], 400)

    def test_request_size_boundary_is_checked_before_reading_oversized_body(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.putrequest("POST", "/api/v1/model/load")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(app.BODY_LIMIT + 1))
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 413)
            self.assertIn(b"8 MiB", response.read())
        finally:
            connection.close()
        base = json.dumps({"model": 3, "padding": ""})
        body = json.dumps({"model": 3, "padding": "x" * (app.BODY_LIMIT - len(base))})
        self.assertEqual(len(body.encode()), app.BODY_LIMIT)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("POST", "/api/v1/model/load", body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn(b"local GGUF", response.read())
        finally:
            connection.close()
        self.assertIsNone(self.runtime.process)

    @unittest.skipIf(sys.platform == "win32", "Windows TerminateProcess does not deliver a Python SIGTERM")
    def test_launcher_shuts_down_on_sigterm(self):
        process = subprocess.Popen(
            [sys.executable, str(APP_PATH), "--port", "0", "--no-browser",
             "--llama-server", str(self.backend), "--execution-mode", "standard"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            line = process.stdout.readline()
            self.assertIn("http://127.0.0.1:", line)
            port = urlsplit(line.split(" ", 1)[1].strip()).port
            backend_pid = None
            client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                client.request("POST", "/api/v1/model/load",
                               json.dumps({"model": str(self.model)}),
                               {"Content-Type": "application/json"})
                response = client.getresponse()
                self.assertEqual(response.status, 202)
                response.read()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    client.request("GET", "/api/v1/status")
                    response = client.getresponse()
                    status = json.loads(response.read())
                    if status["state"] == "ready":
                        backend_pid = json.loads(status["logs"][0])["pid"]
                        break
                    time.sleep(0.03)
                self.assertIsNotNone(backend_pid)
            finally:
                client.close()
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stdout + stderr)
            with self.assertRaises(ProcessLookupError):
                os.kill(backend_pid, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
