"""Smoke the actual zipapp, including resource loading and loopback API."""

import http.client
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from urllib.parse import urlsplit
import zipfile

ARCHIVE = Path(sys.argv.pop(1)).resolve()


class ReleaseTests(unittest.TestCase):
    def test_version_and_public_payload(self):
        result = subprocess.run([sys.executable, str(ARCHIVE), "--version"],
                                capture_output=True, text=True, check=True, timeout=10)
        self.assertIn("SYNAPSE 0.1.0rc4", result.stdout)
        with zipfile.ZipFile(ARCHIVE) as archive:
            names = archive.namelist()
        self.assertIn("synapse_app/web/monitor.mjs", names)
        self.assertIn("synapse_app/tuning.py", names)
        self.assertIn("synapse_app/web/model-controls.mjs", names)
        self.assertFalse(any(name.endswith((".gguf", ".bin", ".log")) or
                             "/results/" in name or "__pycache__" in name for name in names))

    def test_packaged_http_assets_and_monitor(self):
        process = subprocess.Popen(
            [sys.executable, str(ARCHIVE), "--no-browser", "--port", "0", "--execution-mode", "standard",
             "--llama-server", str(ARCHIVE.parent / "missing-backend")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0)
        try:
            line = process.stdout.readline()
            self.assertTrue(line.startswith("SYNAPSE: http://127.0.0.1:"), line)
            port = urlsplit(line.split(" ", 1)[1].strip()).port
            for path, content in (("/", b"SYNAPSE"), ("/monitor.mjs", b"MonitorView"),
                                  ("/chat-input.mjs", b"bindChatInput"), ("/theme.js", b"synapse.theme"),
                                  ("/model-controls.mjs", b"bindReasoningEffort"),
                                  ("/navigation.mjs", b"navigate"), ("/style.css", b"sidebar"),
                                  ("/icon.svg", b"<svg"), ("/api/v1/status", b'"app_name": "SYNAPSE"'),
                                  ("/api/v1/monitor", b'"samples"')):
                with self.subTest(path=path):
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    connection.request("GET", path)
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertIn(content, response.read())
                    connection.close()
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stdout + stderr)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
