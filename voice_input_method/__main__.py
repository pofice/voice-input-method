"""Entry point: python -m voice_input_method [--config path/to/config.yaml]"""

import argparse
import signal
import sys

from PySide6.QtWidgets import QApplication

from .app import MainWindow
from .config import load_config, resolve_resource_path


def main():
    parser = argparse.ArgumentParser(description="Voice Input Method")
    parser.add_argument("--config", "-c", help="Path to config.yaml", default="config.yaml")
    parser.add_argument("--device", "-d", type=int, default=None,
                        help="Audio input device index (see: voice-input-cli devices)")
    args = parser.parse_args()

    config = load_config(args.config)
    config._device_index = args.device

    app = QApplication(sys.argv)

    # Let Ctrl+C kill the app (Qt swallows SIGINT by default)
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    # Load stylesheet
    style_path = resolve_resource_path(config, "style_file")
    if style_path.exists():
        with open(style_path, "r") as f:
            app.setStyleSheet(f.read())

    window = MainWindow(config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
