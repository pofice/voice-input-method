"""Audio recording module with automatic device detection, fallback, and streaming support.

`sounddevice` is imported lazily inside `AudioRecorder.start()` so that
non-recording code paths (CLI, headless tests) don't fail on systems
without the system-level PortAudio library installed.
"""

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import soundfile as sf

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    import sounddevice as sd  # noqa: F401


def resample_to_16k_mono(data: np.ndarray, orig_sr: int, channels: int) -> np.ndarray:
    """Resample audio to 16kHz mono float32 for ASR model input."""
    # Convert to mono if multi-channel
    if data.ndim == 2 and data.shape[1] > 1:
        data = data.mean(axis=1)
    elif data.ndim == 2:
        data = data[:, 0]

    # Resample if needed
    if orig_sr != 16000:
        # Simple linear interpolation resampling (good enough for speech)
        duration = len(data) / orig_sr
        target_len = int(duration * 16000)
        if target_len > 0:
            indices = np.linspace(0, len(data) - 1, target_len)
            data = np.interp(indices, np.arange(len(data)), data)

    return data.astype(np.float32)


class AudioRecorder:
    def __init__(self, sample_rate: int = 44100, channels: int = 2,
                 on_chunk: Callable[[np.ndarray], None] | None = None,
                 chunk_samples: int = 0):
        """
        Args:
            sample_rate: Target recording sample rate.
            channels: Target number of channels.
            on_chunk: Callback for streaming mode. Called with 16kHz mono chunks.
            chunk_samples: Number of 16kHz samples per streaming chunk (0 = no streaming).
        """
        self._target_sample_rate = sample_rate
        self._target_channels = channels
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        # Accumulate numpy chunks (cheap: just a list append of ndarray refs)
        # instead of converting to Python list on every audio callback.
        self.buffer: list[np.ndarray] = []
        self.is_recording: bool = False
        self.stream: Any = None  # sd.InputStream once started

        # Streaming support
        self._on_chunk = on_chunk
        self._chunk_samples_16k = chunk_samples
        self._streaming_buffer: np.ndarray = np.array([], dtype=np.float32)

    def start(self):
        """Initialize and start the audio input stream.

        sounddevice (and its underlying PortAudio system library) is only
        imported here, so headless code paths that never call start() can
        run on systems without PortAudio.
        """
        import sounddevice as sd  # lazy: PortAudio not needed for non-recording flows
        self._sd = sd
        self.sample_rate, self.channels = self._detect_device()
        self.stream = self._open_stream()
        if self.stream:
            self.stream.start()

    def _detect_device(self) -> tuple[int, int]:
        """Detect audio device capabilities and return (sample_rate, channels)."""
        sd = self._sd
        try:
            info = sd.query_devices(kind="input")
            if info:
                max_ch = int(info.get("max_input_channels", 1))
                default_sr = int(info.get("default_samplerate", self._target_sample_rate))
                if max_ch > 0:
                    ch = min(self._target_channels, max_ch)
                    return default_sr, ch
        except Exception as e:
            print(f"Warning: could not query default input device: {e}")

        try:
            for d in sd.query_devices():
                if d.get("max_input_channels", 0) > 0:
                    max_ch = int(d["max_input_channels"])
                    sr = int(d.get("default_samplerate", self._target_sample_rate))
                    return sr, min(self._target_channels, max_ch)
        except Exception as e:
            print(f"Warning: could not enumerate devices: {e}")

        return self._target_sample_rate, 1

    def _open_stream(self):
        """Try to open an InputStream with fallbacks."""
        sd = self._sd
        attempts = [
            {"samplerate": self.sample_rate, "channels": self.channels},
            {"samplerate": self.sample_rate, "channels": 1},
            {},
        ]
        for params in attempts:
            try:
                stream = sd.InputStream(callback=self._audio_callback, **params)
                if params:
                    self.sample_rate = params.get("samplerate", self.sample_rate)
                    self.channels = params.get("channels", self.channels)
                else:
                    self.sample_rate = int(stream.samplerate)
                    self.channels = stream.channels
                return stream
            except Exception as e:
                print(f"Failed to open InputStream with {params}: {e}")
        print("Unable to start audio input stream.")
        return None

    def _audio_callback(self, indata, frames, time, status):
        if not self.is_recording:
            return

        # Copy because sounddevice reuses the buffer between callbacks.
        self.buffer.append(indata.copy())

        # Streaming: resample incoming audio and feed chunks
        if self._on_chunk and self._chunk_samples_16k > 0:
            resampled = resample_to_16k_mono(indata.copy(), self.sample_rate, self.channels)
            self._streaming_buffer = np.concatenate([self._streaming_buffer, resampled])

            while len(self._streaming_buffer) >= self._chunk_samples_16k:
                chunk = self._streaming_buffer[:self._chunk_samples_16k]
                self._streaming_buffer = self._streaming_buffer[self._chunk_samples_16k:]
                self._on_chunk(chunk)

    def start_recording(self):
        self.buffer = []
        self._streaming_buffer = np.array([], dtype=np.float32)
        self.is_recording = True

    def _concat_buffer(self) -> np.ndarray:
        """Concatenate accumulated chunk arrays into one array."""
        if not self.buffer:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(self.buffer, axis=0)

    def stop_recording(self, output_path: str) -> str:
        """Stop recording and save to WAV file. Returns the output path."""
        self.is_recording = False
        if self.buffer:
            data = self._concat_buffer()
            sf.write(output_path, data, self.sample_rate)
        return output_path

    def get_recording_16k_mono(self) -> np.ndarray:
        """Get the full recording resampled to 16kHz mono."""
        if not self.buffer:
            return np.array([], dtype=np.float32)
        data = self._concat_buffer()
        return resample_to_16k_mono(data, self.sample_rate, self.channels)

    def flush_streaming_buffer(self):
        """Flush any remaining audio in the streaming buffer."""
        if self._on_chunk and len(self._streaming_buffer) > 0:
            self._on_chunk(self._streaming_buffer)
            self._streaming_buffer = np.array([], dtype=np.float32)

    def switch_device(self, device_index: int | None = None) -> None:
        """Switch to a different input device at runtime.

        Args:
            device_index: sounddevice device index, or None for system default.
        """
        sd = self._sd
        # Stop current stream
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        # Query new device
        try:
            if device_index is not None:
                info = sd.query_devices(device_index)
                max_ch = int(info.get("max_input_channels", 1))
                self.sample_rate = int(info.get("default_samplerate", self._target_sample_rate))
                self.channels = min(self._target_channels, max_ch)
            else:
                self.sample_rate, self.channels = self._detect_device()
                device_index = None
        except Exception as e:
            print(f"Warning: failed to query device {device_index}: {e}")
            self.sample_rate, self.channels = self._detect_device()
            device_index = None

        # Open new stream
        try:
            self.stream = sd.InputStream(
                device=device_index,
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as e:
            print(f"Failed to open device {device_index}: {e}, falling back")
            self.stream = self._open_stream()
            if self.stream:
                self.stream.start()

    @staticmethod
    def list_input_devices(refresh: bool = True) -> list[dict]:
        """Return a list of available input devices (re-scans hardware)."""
        import sounddevice as sd
        if refresh:
            # Force PortAudio to re-scan devices (it caches the list).
            # Existing streams survive re-init on macOS/Windows.
            sd._terminate()
            sd._initialize()
        devices = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                devices.append({
                    "index": i,
                    "name": d["name"],
                    "channels": d["max_input_channels"],
                    "sample_rate": int(d.get("default_samplerate", 44100)),
                })
        return devices

    def stop(self):
        """Stop the audio stream."""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
