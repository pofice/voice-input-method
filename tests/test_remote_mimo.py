"""Unit tests for the remote-mimo backend against a stub Gradio server.

Spins up a local http.server that mimics the three Gradio 5 endpoints
(upload / call / SSE result) so the full HTTP round trip is exercised
without a real GPU server.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from voice_input_method.recognition.remote_mimo import RemoteMiMoRecognizer


class _StubGradioHandler(BaseHTTPRequestHandler):
    result_text = "你好，世界。"
    fail_event = False

    def log_message(self, *args):  # silence test output
        pass

    def _send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/gradio_api/info":
            self._send_json({"named_endpoints": {"/transcribe": {}}})
        elif self.path.startswith("/gradio_api/call/transcribe/"):
            if self.fail_event:
                body = b'event: error\ndata: "boom"\n\n'
            else:
                data = json.dumps([self.result_text, "Transcription completed"])
                body = f"event: complete\ndata: {data}\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/gradio_api/upload":
            self._send_json(["/tmp/gradio/fake.wav"])
        elif self.path == "/gradio_api/call/transcribe":
            self._send_json({"event_id": "ev123"})
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def stub_server():
    server = HTTPServer(("127.0.0.1", 0), _StubGradioHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_transcribe_round_trip(stub_server, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfakewav")
    rec = RemoteMiMoRecognizer(base_url=stub_server, timeout=5)
    rec.load()
    assert rec.transcribe(str(wav)) == "你好，世界。"


def test_server_error_event_raises(stub_server, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfakewav")
    _StubGradioHandler.fail_event = True
    try:
        rec = RemoteMiMoRecognizer(base_url=stub_server, timeout=5)
        with pytest.raises(RuntimeError, match="server error"):
            rec.transcribe(str(wav))
    finally:
        _StubGradioHandler.fail_event = False


def test_load_unreachable_raises():
    rec = RemoteMiMoRecognizer(base_url="http://127.0.0.1:1", timeout=1)
    with pytest.raises(RuntimeError, match="unreachable"):
        rec.load()


def test_warmup_swallows_errors():
    rec = RemoteMiMoRecognizer(base_url="http://127.0.0.1:1", timeout=1)
    rec.warmup("/nonexistent.wav")  # must not raise
