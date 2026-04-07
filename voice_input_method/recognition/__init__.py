"""Speech recognition backends.

Each backend implements the Recognizer protocol (protocols.py).
Factory selects the backend based on config.recognizer_backend.
"""

from .funasr_recognizer import FunASRRecognizer, FunASRStreamingRecognizer

__all__ = [
    "FunASRRecognizer",
    "FunASRStreamingRecognizer",
]
