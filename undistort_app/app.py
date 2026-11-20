from __future__ import annotations

import ctypes
from pathlib import Path
import sys
from typing import Sequence

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow
from .ui.styles import apply_styles


def _set_windows_app_id() -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Intrinsic.PhoneUndistort.App")
        except Exception:
            pass


def main(argv: Sequence[str] | None = None) -> None:
    _set_windows_app_id()
    args = list(sys.argv if argv is None else argv)
    app = QApplication(args)
    QCoreApplication.setOrganizationName("Intrinsic")
    app.setApplicationName("PNG Undistortion")
    apply_styles(app)

    icon_path = Path(__file__).resolve().parent / "icon.ico"
    app_icon = QIcon(str(icon_path)) if icon_path.exists() else QIcon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    initial_files = [arg for arg in args[1:] if arg and not arg.startswith("-")]
    window = MainWindow(initial_files=initial_files)
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
