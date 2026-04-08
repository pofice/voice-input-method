"""Speech recognition backends.

Each backend implements the Recognizer protocol (protocols.py).
Factory selects the backend based on config.recognizer_backend.

Backends are NOT eagerly imported here because each has heavy,
optional dependencies (funasr_onnx / sherpa_onnx). Import directly
from the submodule you need, or let factory._create_recognizer() do it.
"""

__all__: list[str] = []
