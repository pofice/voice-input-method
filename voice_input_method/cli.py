"""Headless CLI entry point for voice-input-method.

Designed for AI agents and CI/CD: take a WAV file in, get text out.
No GUI, no microphone, no hotkeys.

Usage::

    voice-input-cli transcribe input.wav
    voice-input-cli transcribe input.wav --output result.txt
    voice-input-cli transcribe input.wav --streaming
    voice-input-cli transcribe input.wav --hotwords "遍历 数组 函数"
    voice-input-cli transcribe input.wav --json
    voice-input-cli batch ./audio_dir/ --output results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import load_config, DEFAULT_OFFLINE_MODELS, DEFAULT_STREAMING_MODEL
from .recognition.funasr_recognizer import FunASRRecognizer, FunASRStreamingRecognizer
from .text_processing import clean_spaces


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Transcribe a single WAV file."""
    wav_path = Path(args.input)
    if not wav_path.exists():
        print(f"Error: file not found: {wav_path}", file=sys.stderr)
        return 1

    hotwords = args.hotwords or ""
    start = time.time()

    if args.streaming:
        result_text, partials = _transcribe_streaming(
            wav_path,
            model_dir=args.model or DEFAULT_STREAMING_MODEL,
            quantize=not args.no_quantize,
        )
    else:
        result_text = _transcribe_offline(
            wav_path,
            model_dir=args.model or DEFAULT_OFFLINE_MODELS["seaco_paraformer"],
            quantize=not args.no_quantize,
            hotwords=hotwords,
        )
        partials = []

    elapsed_ms = int((time.time() - start) * 1000)

    if args.json:
        output = {
            "input": str(wav_path),
            "text": result_text,
            "elapsed_ms": elapsed_ms,
            "mode": "streaming" if args.streaming else "offline",
        }
        if args.streaming:
            output["partials"] = partials
        text_out = json.dumps(output, ensure_ascii=False, indent=2)
    else:
        text_out = result_text

    if args.output:
        Path(args.output).write_text(text_out + "\n", encoding="utf-8")
        print(f"Wrote {args.output} ({elapsed_ms} ms)", file=sys.stderr)
    else:
        print(text_out)

    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Transcribe a directory of WAV files, output JSONL."""
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        return 1

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"Error: no .wav files in {input_dir}", file=sys.stderr)
        return 1

    print(f"Loading model...", file=sys.stderr)
    recognizer = FunASRRecognizer(
        model_dir=args.model or DEFAULT_OFFLINE_MODELS["seaco_paraformer"],
        quantize=not args.no_quantize,
    )
    recognizer.load()

    output_path = Path(args.output) if args.output else None
    out_fp = open(output_path, "w", encoding="utf-8") if output_path else sys.stdout

    try:
        for i, wav in enumerate(wav_files, 1):
            print(f"[{i}/{len(wav_files)}] {wav.name}", file=sys.stderr)
            start = time.time()
            text = recognizer.transcribe(str(wav), args.hotwords or "")
            text = clean_spaces(text)
            elapsed_ms = int((time.time() - start) * 1000)

            record = {
                "file": wav.name,
                "text": text,
                "elapsed_ms": elapsed_ms,
            }
            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fp.flush()
    finally:
        if output_path:
            out_fp.close()
            print(f"Wrote {output_path}", file=sys.stderr)

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Print version and environment info."""
    from . import __version__
    info = {
        "version": __version__,
        "default_offline_model": DEFAULT_OFFLINE_MODELS["seaco_paraformer"],
        "default_streaming_model": DEFAULT_STREAMING_MODEL,
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """End-to-end self-test: verify deps, download model, run inference.

    Designed to be the single command an AI agent runs after `pip install`
    to confirm the project is fully usable. Prints structured progress so
    the agent can detect partial failures.

    Exit code 0 = healthy, non-zero = something is broken.
    """
    import importlib
    import tempfile

    checks: list[dict] = []
    overall_ok = True

    def step(name: str, fn):
        """Run a single check, capture result."""
        nonlocal overall_ok
        print(f"  [..] {name}", file=sys.stderr, flush=True)
        try:
            detail = fn()
            checks.append({"check": name, "status": "ok", "detail": detail})
            print(f"\r  [OK] {name}: {detail}", file=sys.stderr, flush=True)
            return True
        except Exception as e:
            checks.append({"check": name, "status": "fail", "error": str(e)})
            print(f"\r  [FAIL] {name}: {e}", file=sys.stderr, flush=True)
            overall_ok = False
            return False

    # 1. Core imports
    def check_imports():
        modules = [
            "voice_input_method.engine",
            "voice_input_method.factory",
            "voice_input_method.recognition",
            "voice_input_method.text_processing",
            "voice_input_method.cli",
            "funasr_onnx",
            "soundfile",
            "numpy",
        ]
        for m in modules:
            importlib.import_module(m)
        return f"{len(modules)} modules importable"

    # 2. Build a tiny test WAV (1s of silence)
    test_wav_path = None

    def check_audio_io():
        nonlocal test_wav_path
        import numpy as np
        import soundfile as sf
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, np.zeros(16000, dtype=np.float32), 16000)
        # Read back to confirm
        data, sr = sf.read(tmp.name, dtype="float32")
        assert sr == 16000 and len(data) == 16000
        test_wav_path = tmp.name
        return "16kHz mono WAV write+read OK"

    # 3. Model download + load
    recognizer_holder = {}

    def check_model_load():
        rec = FunASRRecognizer(
            model_dir=args.model or DEFAULT_OFFLINE_MODELS["seaco_paraformer"],
            quantize=not args.no_quantize,
        )
        rec.load()
        recognizer_holder["rec"] = rec
        model_id = args.model or DEFAULT_OFFLINE_MODELS["seaco_paraformer"]
        return f"loaded {model_id}"

    # 4. Run inference (silence is enough — we just need it not to crash)
    def check_inference():
        rec = recognizer_holder["rec"]
        text = rec.transcribe(test_wav_path)
        # Doesn't matter what comes back; success means the pipeline works.
        return f"inference returned {len(text)} chars"

    # 5. Optionally test with real audio if a fixture exists
    def check_real_audio():
        from pathlib import Path
        # Look for a test fixture in the repo (when run from source checkout)
        candidates = [
            Path(__file__).parent.parent / "tests" / "fixtures" / "chinese_speech_16k.wav",
            Path.cwd() / "tests" / "fixtures" / "chinese_speech_16k.wav",
        ]
        fixture = next((p for p in candidates if p.exists()), None)
        if fixture is None:
            return "no fixture found (skipped)"
        rec = recognizer_holder["rec"]
        from .text_processing import clean_spaces
        text = clean_spaces(rec.transcribe(str(fixture)))
        # Expected content for the bundled fixture
        expected = ["今天", "天气", "公园"]
        missing = [k for k in expected if k not in text]
        if missing:
            raise RuntimeError(f"missing keywords {missing} in: {text!r}")
        return f"recognized {text!r}"

    print("voice-input-method doctor", file=sys.stderr)
    step("import core dependencies", check_imports)
    step("audio I/O", check_audio_io)
    step("ASR model load", check_model_load)
    step("ASR inference (silence)", check_inference)
    step("real Chinese audio", check_real_audio)

    # Cleanup
    if test_wav_path:
        try:
            Path(test_wav_path).unlink()
        except OSError:
            pass

    result = {
        "ok": overall_ok,
        "checks": checks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        f"\n{'OK — ready to use' if overall_ok else 'FAILED — see checks above'}",
        file=sys.stderr,
    )
    return 0 if overall_ok else 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _transcribe_offline(
    wav_path: Path, model_dir: str, quantize: bool, hotwords: str
) -> str:
    print(f"Loading offline model: {model_dir}", file=sys.stderr)
    recognizer = FunASRRecognizer(
        model_dir=model_dir,
        quantize=quantize,
    )
    recognizer.load()
    text = recognizer.transcribe(str(wav_path), hotwords)
    return clean_spaces(text)


def _transcribe_streaming(
    wav_path: Path, model_dir: str, quantize: bool
) -> tuple[str, list[str]]:
    import numpy as np
    import soundfile as sf

    print(f"Loading streaming model: {model_dir}", file=sys.stderr)
    recognizer = FunASRStreamingRecognizer(
        model_dir=model_dir,
        quantize=quantize,
        chunk_size=[5, 10, 5],
    )
    recognizer.load()

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        print(
            f"Warning: input is {sr} Hz, streaming model expects 16 kHz",
            file=sys.stderr,
        )

    recognizer.reset()
    step = recognizer.step_samples
    offset = 0
    full_text = ""
    partials: list[str] = []

    while offset < len(audio):
        remaining = len(audio) - offset
        is_final = remaining <= step
        chunk = audio[offset : offset + min(step, remaining)]
        text = recognizer.feed_chunk(chunk, is_final=is_final)
        if text:
            full_text += text
            partials.append(text)
        offset += step

    return clean_spaces(full_text), partials


# ---------------------------------------------------------------------------
# Argparse setup
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-input-cli",
        description="Headless CLI for voice-input-method (no GUI required).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # transcribe
    p_t = sub.add_parser("transcribe", help="Transcribe a single WAV file")
    p_t.add_argument("input", help="Path to input .wav file")
    p_t.add_argument("-o", "--output", help="Write result to file (default: stdout)")
    p_t.add_argument(
        "-m", "--model", help="Model dir or ModelScope ID (default: built-in)"
    )
    p_t.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming Paraformer-online instead of offline",
    )
    p_t.add_argument(
        "--hotwords",
        help="Space-separated hotwords (offline mode, seaco_paraformer only)",
    )
    p_t.add_argument(
        "--no-quantize", action="store_true", help="Disable INT8 quantization"
    )
    p_t.add_argument(
        "--json", action="store_true", help="Output JSON with metadata instead of text"
    )
    p_t.set_defaults(func=cmd_transcribe)

    # batch
    p_b = sub.add_parser("batch", help="Transcribe a directory of WAV files (JSONL out)")
    p_b.add_argument("input", help="Directory containing .wav files")
    p_b.add_argument("-o", "--output", help="Output JSONL file (default: stdout)")
    p_b.add_argument("-m", "--model", help="Model dir or ModelScope ID")
    p_b.add_argument("--hotwords", help="Space-separated hotwords")
    p_b.add_argument("--no-quantize", action="store_true")
    p_b.set_defaults(func=cmd_batch)

    # info
    p_i = sub.add_parser("info", help="Show version and default model IDs")
    p_i.set_defaults(func=cmd_info)

    # doctor
    p_d = sub.add_parser(
        "doctor",
        help="End-to-end self-test (deps + model download + inference)",
    )
    p_d.add_argument("-m", "--model", help="Model ID to test (default: built-in)")
    p_d.add_argument("--no-quantize", action="store_true")
    p_d.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
