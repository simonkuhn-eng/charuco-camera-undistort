from PySide6.QtWidgets import QApplication


_QSS = """
QWidget {
    background: #f7f3ee;
    color: #2b2a28;
    font-family: "Poppins", "Segoe UI";
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #d8d2c8;
    border-radius: 10px;
    margin-top: 10px;
    padding: 12px;
    background: #fbf8f3;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px 0 4px;
    color: #5b5550;
}

QPushButton {
    background: #0b8b8b;
    color: #ffffff;
    border-radius: 8px;
    padding: 8px 12px;
    font-weight: 500;
}

QPushButton:hover {
    background: #097575;
}

QPushButton:pressed {
    background: #075f5f;
}

QPushButton:disabled {
    background: #a9c9c9;
    color: #f4f4f4;
}

QPushButton[secondary="true"] {
    background: #eae3d8;
    color: #2b2a28;
    border: 1px solid #d0c8bc;
}

QPushButton[secondary="true"]:hover {
    background: #dfd6c8;
}

QPushButton[secondary="true"]:disabled {
    background: #f0ece5;
    color: #9e978e;
    border: 1px solid #e2ddd4;
}

QLabel#dropBanner {
    background: #eef6f6;
    color: #0b6e6e;
    border: 1px dashed #0b8b8b;
    border-radius: 8px;
    padding: 8px 10px;
    font-weight: 500;
}

QLabel#statusInfo {
    color: #5b5550;
    font-size: 12px;
}

QLabel[previewFrame="true"] {
    background: #f0ebe3;
    border: 1px solid #dcd4c8;
    border-radius: 6px;
    padding: 4px;
}

QLineEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox,
QListWidget,
QPlainTextEdit {
    background: #fffdf8;
    border: 1px solid #d8d2c8;
    border-radius: 6px;
    padding: 4px;
}

QTabWidget::pane {
    border: 0;
}

QTabBar::tab {
    background: #e7e0d6;
    padding: 8px 12px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 4px;
}

QTabBar::tab:selected {
    background: #fffdf8;
}
"""


def apply_styles(app: QApplication) -> None:
    app.setStyleSheet(_QSS)
