"""Audio recording module with automatic device detection and fallback."""

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        self._target_sample_rate = sample_rate
        self._target_channels = channels
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self.buffer: list = []
        self.is_recording: bool = False
        self.stream: sd.InputStream | None = None

    def start(self):
        """Initialize and start the audio input stream with device detection and fallback."""
        self.sample_rate, self.channels = self._detect_device()
        self.stream = self._open_stream()
        if self.stream:
            self.stream.start()

    def _detect_device(self) -> tuple[int, int]:
        """Detect audio device capabilities and return (sample_rate, channels)."""
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

        # Try to find any input device
        try:
            for d in sd.query_devices():
                if d.get("max_input_channels", 0) > 0:
                    max_ch = int(d["max_input_channels"])
                    sr = int(d.get("default_samplerate", self._target_sample_rate))
                    return sr, min(self._target_channels, max_ch)
        except Exception as e:
            print(f"Warning: could not enumerate devices: {e}")

        return self._target_sample_rate, 1

    def _open_stream(self) -> sd.InputStream | None:
        """Try to open an InputStream with fallbacks."""
        attempts = [
            {"samplerate": self.sample_rate, "channels": self.channels},
            {"samplerate": self.sample_rate, "channels": 1},
            {},  # Let sounddevice pick defaults
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
        if self.is_recording:
            self.buffer.extend(indata.tolist())

    def start_recording(self):
        self.buffer = []
        self.is_recording = True

    def stop_recording(self, output_path: str) -> str:
        """Stop recording and save to WAV file. Returns the output path."""
        self.is_recording = False
        if self.buffer:
            data = np.array(self.buffer)
            sf.write(output_path, data, self.sample_rate)
        return output_path

    def stop(self):
        """Stop the audio stream."""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
