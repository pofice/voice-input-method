"""Remote MiMo-V2.5-ASR backend via its Gradio HTTP API.

Talks to a MiMo-V2.5-ASR Gradio server (run_mimo_asr.py from
https://github.com/XiaomiMiMo/MiMo-V2.5-ASR) over the LAN. All heavy
inference happens on the server GPU; this client only does HTTP, so it
adds no local dependencies (stdlib urllib only).

Gradio 5 API flow:
  1. POST {base}/gradio_api/upload            (multipart) -> [server_path]
  2. POST {base}/gradio_api/call/transcribe   (json)      -> {event_id}
  3. GET  {base}/gradio_api/call/transcribe/{event_id}    -> SSE, final
     "data:" line carries [transcription, status].
"""

from __future__ import annotations

import json
import urllib.request
import uuid
from pathlib import Path


class RemoteMiMoRecognizer:
    """Offline speech recognizer backed by a remote MiMo-V2.5-ASR server."""

    def __init__(
        self,
        base_url: str,
        language: str = "Auto",
        timeout: float = 60.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._timeout = timeout
        # Connect directly, ignoring HTTP(S)_PROXY env vars: the server is
        # on the LAN and an external proxy cannot route to it (502s).
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )

    def load(self) -> None:
        """Verify the server is reachable and the app is up."""
        req = urllib.request.Request(f"{self._base_url}/gradio_api/info")
        try:
            with self._opener.open(req, timeout=self._timeout) as resp:
                resp.read()
        except OSError as exc:
            raise RuntimeError(
                f"MiMo ASR server unreachable at {self._base_url}: {exc}"
            ) from exc

    def warmup(self, warmup_wav: str, hotwords: str = "") -> None:
        """One round trip to warm server caches; failures are non-fatal."""
        if not warmup_wav:
            return
        try:
            self.transcribe(warmup_wav)
        except Exception:
            pass

    def transcribe(self, wav_path: str, hotwords: str = "") -> str:
        # hotwords are unsupported by the MiMo API and ignored here;
        # post-ASR correction rules in hotwords.txt still apply downstream.
        server_path = self._upload(wav_path)
        event_id = self._call(server_path)
        return self._result(event_id)

    def _upload(self, wav_path: str) -> str:
        path = Path(wav_path)
        boundary = uuid.uuid4().hex
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="files"; '
                f'filename="{path.name}"\r\n'.encode(),
                b"Content-Type: audio/wav\r\n\r\n",
                path.read_bytes(),
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        req = urllib.request.Request(
            f"{self._base_url}/gradio_api/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with self._opener.open(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())[0]

    def _call(self, server_path: str) -> str:
        payload = json.dumps(
            {
                "data": [
                    {"path": server_path, "meta": {"_type": "gradio.FileData"}},
                    None,
                    self._language,
                ]
            }
        ).encode()
        req = urllib.request.Request(
            f"{self._base_url}/gradio_api/call/transcribe",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with self._opener.open(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())["event_id"]

    def _result(self, event_id: str) -> str:
        req = urllib.request.Request(
            f"{self._base_url}/gradio_api/call/transcribe/{event_id}"
        )
        with self._opener.open(req, timeout=self._timeout) as resp:
            event = ""
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    if event == "error":
                        raise RuntimeError(f"MiMo ASR server error: {line[5:].strip()}")
                    if event == "complete":
                        data = json.loads(line[5:])
                        return (data[0] or "").strip()
        raise RuntimeError("MiMo ASR server closed the stream without a result")
