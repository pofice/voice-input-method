"""Entry point: python -m voice_input_method [--config path/to/config.yaml]"""

import argparse
import sys

from PySide6.QtWidgets import QApplication

from .config import load_config, resolve_resource_path
from .app import MainWindow


def main():
    parser = argparse.ArgumentParser(description="Voice Input Method")
    parser.add_argument("--config", "-c", help="Path to config.yaml", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    app = QApplication(sys.argv)

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
