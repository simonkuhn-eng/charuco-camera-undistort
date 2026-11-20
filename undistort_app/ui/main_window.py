from __future__ import annotations

from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.calibration import calibrate_charuco, read_image
from ..core.patterns import PatternConfig
from ..core.persistence import CalibrationData, load_calibration_json, save_calibration_json
from ..core.undistort import adapt_calibration_for_size, undistort_image, write_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
IMAGE_FILE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"


def cv_to_qimage(image: np.ndarray) -> QImage:
    """Convert an OpenCV BGR/BGRA/Gray NumPy array to a standalone QImage."""
    if image.ndim == 2:
        gray = np.ascontiguousarray(image)
        height, width = gray.shape
        qimage = QImage(gray.data, width, height, gray.strides[0], QImage.Format_Grayscale8)
        return qimage.copy()

    if image.shape[2] == 4:
        rgba = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        height, width, _ = rgba.shape
        qimage = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format_RGBA8888)
        return qimage.copy()

    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    height, width, _ = rgb.shape
    qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888)
    return qimage.copy()


def qimage_to_cv(qimage: QImage) -> np.ndarray:
    """Convert a QImage into an OpenCV BGR or BGRA NumPy array."""
    if qimage.hasAlphaChannel():
        converted = qimage.convertToFormat(QImage.Format_RGBA8888)
        width, height, bpl = converted.width(), converted.height(), converted.bytesPerLine()
        buf = np.frombuffer(converted.constBits(), dtype=np.uint8, count=height * bpl)
        rgba = buf.reshape((height, bpl))[:, : width * 4].reshape((height, width, 4))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

    converted = qimage.convertToFormat(QImage.Format_RGB888)
    width, height, bpl = converted.width(), converted.height(), converted.bytesPerLine()
    buf = np.frombuffer(converted.constBits(), dtype=np.uint8, count=height * bpl)
    rgb = buf.reshape((height, bpl))[:, : width * 3].reshape((height, width, 3))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def scale_pixmap_to_label(pixmap: QPixmap, label: QLabel) -> QPixmap:
    """Scale a QPixmap smoothly to fit within a QLabel while keeping aspect ratio."""
    if pixmap.isNull():
        return pixmap

    target_size = label.size()
    if target_size.width() < 2 or target_size.height() < 2:
        hint = label.sizeHint()
        target_size = QSize(max(hint.width(), 360), max(hint.height(), 240))

    return pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class ClickableLabel(QLabel):
    """QLabel that emits a clicked signal when left-clicked."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class BeforeAfterPreview(QWidget):
    """Reusable side-by-side or clickable single-image Before/After preview."""

    def __init__(self, allow_single_toggle: bool = False) -> None:
        super().__init__()
        self._before_image: Optional[np.ndarray] = None
        self._after_image: Optional[np.ndarray] = None
        self._show_after_single = True
        self._allow_single_toggle = allow_single_toggle

        self.before_title = QLabel("Vorher (Original)")
        self.before_title.setAlignment(Qt.AlignCenter)

        self.after_title = QLabel("Nachher (Entzerrt)")
        self.after_title.setAlignment(Qt.AlignCenter)

        self.before_label = self._make_preview_label("Keine Vorschau")
        self.after_label = self._make_preview_label("Keine Vorschau")

        self.before_panel = QWidget()
        before_layout = QVBoxLayout(self.before_panel)
        before_layout.setContentsMargins(0, 0, 0, 0)
        before_layout.addWidget(self.before_title)
        before_layout.addWidget(self.before_label)

        self.after_panel = QWidget()
        after_layout = QVBoxLayout(self.after_panel)
        after_layout.setContentsMargins(0, 0, 0, 0)
        after_layout.addWidget(self.after_title)
        after_layout.addWidget(self.after_label)

        panels_layout = QHBoxLayout()
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panels_layout.addWidget(self.before_panel)
        panels_layout.addWidget(self.after_panel)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self.single_mode_check: Optional[QCheckBox] = None
        if allow_single_toggle:
            self.single_mode_check = QCheckBox("Einzelbild-Vergleich (Klicken zum Umschalten)")
            self.single_mode_check.toggled.connect(self._on_mode_toggled)
            root_layout.addWidget(self.single_mode_check)
            self.before_label.clicked.connect(self._toggle_single_view)
            self.after_label.clicked.connect(self._toggle_single_view)

        root_layout.addLayout(panels_layout)

    @staticmethod
    def _make_preview_label(placeholder: str) -> ClickableLabel:
        label = ClickableLabel(placeholder)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(230)
        label.setMinimumWidth(240)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setProperty("previewFrame", True)
        return label

    @property
    def before_image(self) -> Optional[np.ndarray]:
        return self._before_image

    @property
    def after_image(self) -> Optional[np.ndarray]:
        return self._after_image

    def set_pair(
        self,
        before_image: Optional[np.ndarray],
        after_image: Optional[np.ndarray],
    ) -> None:
        self._before_image = before_image
        self._after_image = after_image
        self._show_after_single = True
        self._update_visibility()
        self.refresh()
        QTimer.singleShot(0, self.refresh)

    def clear_pair(self) -> None:
        self._before_image = None
        self._after_image = None
        self.before_label.clear()
        self.before_label.setText("Keine Vorschau")
        self.after_label.clear()
        self.after_label.setText("Keine Vorschau")

    def _is_single_mode(self) -> bool:
        return self.single_mode_check is not None and self.single_mode_check.isChecked()

    def _on_mode_toggled(self) -> None:
        self._show_after_single = True
        self._update_visibility()
        self.refresh()
        QTimer.singleShot(0, self.refresh)

    def _toggle_single_view(self) -> None:
        if not self._is_single_mode():
            return
        if self._before_image is None and self._after_image is None:
            return
        self._show_after_single = not self._show_after_single
        self._update_visibility()
        self.refresh()

    def _update_visibility(self) -> None:
        if not self._is_single_mode():
            self.before_panel.setVisible(True)
            self.after_panel.setVisible(True)
            self.before_title.setText("Vorher (Original)")
            self.after_title.setText("Nachher (Entzerrt)")
            return

        show_after = self._show_after_single
        self.before_panel.setVisible(not show_after)
        self.after_panel.setVisible(show_after)
        self.before_title.setText("Vorher (Original) – Klicken für Entzerrt")
        self.after_title.setText("Nachher (Entzerrt) – Klicken für Original")

    def refresh(self) -> None:
        if self._before_image is not None:
            pixmap = QPixmap.fromImage(cv_to_qimage(self._before_image))
            self.before_label.setPixmap(scale_pixmap_to_label(pixmap, self.before_label))
        else:
            self.before_label.setPixmap(QPixmap())
            self.before_label.setText("Keine Vorschau")

        if self._after_image is not None:
            pixmap = QPixmap.fromImage(cv_to_qimage(self._after_image))
            self.after_label.setPixmap(scale_pixmap_to_label(pixmap, self.after_label))
        else:
            self.after_label.setPixmap(QPixmap())
            self.after_label.setText("Keine Vorschau")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.refresh()


class MainWindow(QMainWindow):
    def __init__(self, initial_files: Optional[List[str]] = None) -> None:
        super().__init__()

        self.setWindowTitle("Phone Undistortion – Mini Desktop App")
        self.resize(1160, 760)

        self.undistort_tab = UndistortTab()
        self.calibration_tab = CalibrationTab()
        self.calibration_tab.calibration_saved.connect(self.undistort_tab.on_calibration_saved)

        tabs = QTabWidget()
        tabs.addTab(self.undistort_tab, "Entzerren (Quick)")
        tabs.addTab(self.calibration_tab, "Kalibrierung")
        self.setCentralWidget(tabs)

        if initial_files:
            QTimer.singleShot(0, lambda: self.undistort_tab.handle_paths(initial_files))


class UndistortTab(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.settings = QSettings()
        self.setAcceptDrops(True)

        drop_banner = QLabel(
            "Bilder oder Ordner hierher ziehen (Drag & Drop)  •  Strg+V aus Zwischenablage"
        )
        drop_banner.setObjectName("dropBanner")
        drop_banner.setAlignment(Qt.AlignCenter)

        self.calibration_path = QLineEdit()
        self.calibration_path.editingFinished.connect(self._on_calibration_edited)
        calib_button = QPushButton("Durchsuchen")
        calib_button.setProperty("secondary", True)
        calib_button.clicked.connect(self._browse_calibration)

        calib_row = QHBoxLayout()
        calib_row.addWidget(self.calibration_path)
        calib_row.addWidget(calib_button)

        calib_group = QGroupBox("Kalibrierung (JSON)")
        calib_layout = QVBoxLayout()
        calib_layout.addLayout(calib_row)
        calib_group.setLayout(calib_layout)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.image_list.currentItemChanged.connect(self._on_selected_image_changed)

        add_button = QPushButton("Bilder wählen")
        add_button.clicked.connect(self._add_images)

        remove_button = QPushButton("Auswahl entfernen")
        remove_button.setProperty("secondary", True)
        remove_button.clicked.connect(self._remove_selected)

        clear_button = QPushButton("Leeren")
        clear_button.setProperty("secondary", True)
        clear_button.clicked.connect(self._clear_images)

        image_buttons = QHBoxLayout()
        image_buttons.addWidget(add_button)
        image_buttons.addWidget(remove_button)
        image_buttons.addWidget(clear_button)

        paste_button = QPushButton("Aus Zwischenablage einfügen (Strg+V)")
        paste_button.setProperty("secondary", True)
        paste_button.clicked.connect(self._paste_from_clipboard)

        images_group = QGroupBox("Eingabebilder")
        images_layout = QVBoxLayout()
        images_layout.addWidget(self.image_list)
        images_layout.addLayout(image_buttons)
        images_layout.addWidget(paste_button)
        images_group.setLayout(images_layout)

        self.output_dir = QLineEdit()
        self.output_dir.editingFinished.connect(self._save_settings)
        output_button = QPushButton("Ordner")
        output_button.setProperty("secondary", True)
        output_button.clicked.connect(self._browse_output_dir)

        open_output_button = QPushButton("Ausgabeordner öffnen")
        open_output_button.setProperty("secondary", True)
        open_output_button.clicked.connect(self._open_output_dir)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir)
        output_row.addWidget(output_button)

        output_group = QGroupBox("Ausgabeordner")
        output_layout = QVBoxLayout()
        output_layout.addLayout(output_row)
        output_layout.addWidget(open_output_button)
        output_group.setLayout(output_layout)

        self.keep_border = QCheckBox("Rand behalten (kein Zuschnitt)")
        self.keep_border.setChecked(True)
        self.keep_border.toggled.connect(self._on_option_toggled)

        self.auto_run = QCheckBox("Sofort beim Ablegen entzerren (Auto-Run)")
        self.auto_run.setChecked(True)
        self.auto_run.toggled.connect(self._save_settings)

        self.run_button = QPushButton("Alle entzerren")
        self.run_button.clicked.connect(self._run_undistort)

        left_layout = QVBoxLayout()
        left_layout.addWidget(drop_banner)
        left_layout.addWidget(calib_group)
        left_layout.addWidget(images_group)
        left_layout.addWidget(output_group)
        left_layout.addWidget(self.keep_border)
        left_layout.addWidget(self.auto_run)
        left_layout.addWidget(self.run_button)

        self.preview_widget = BeforeAfterPreview(allow_single_toggle=True)
        self.adaptation_label = QLabel("Bereit – Bild ablegen oder auswählen.")
        self.adaptation_label.setObjectName("statusInfo")

        self.copy_result_button = QPushButton("Ergebnis in Zwischenablage kopieren")
        self.copy_result_button.setProperty("secondary", True)
        self.copy_result_button.setEnabled(False)
        self.copy_result_button.clicked.connect(self._copy_result_to_clipboard)

        preview_group = QGroupBox("Vorschau (Vorher / Nachher)")
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.preview_widget)
        preview_layout.addWidget(self.adaptation_label)
        preview_layout.addWidget(self.copy_result_button)
        preview_group.setLayout(preview_layout)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)

        log_group = QGroupBox("Protokoll")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log)
        log_group.setLayout(log_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(preview_group, 3)
        right_layout.addWidget(log_group, 1)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 3)
        self.setLayout(main_layout)

        paste_shortcut = QShortcut(QKeySequence.Paste, self)
        paste_shortcut.activated.connect(self._paste_from_clipboard)

        self._load_settings()

    def dragEnterEvent(self, event) -> None:
        mime = event.mimeData()
        if mime.hasUrls() or mime.hasImage():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        mime = event.mimeData()
        if mime.hasUrls():
            local_paths = [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]
            if local_paths:
                self.handle_paths(local_paths)
                event.acceptProposedAction()
                return
        if mime.hasImage():
            qimage = QImage(mime.imageData())
            if not qimage.isNull():
                self._process_in_memory_image(qimage, source_label="dragdrop")
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def on_calibration_saved(self, path: str) -> None:
        self.calibration_path.setText(path)
        self._save_settings()
        self._log(f"Neue Kalibrierung übernommen: {path}")

    def handle_paths(self, raw_paths: Iterable[str | Path]) -> None:
        existing = {self.image_list.item(i).text() for i in range(self.image_list.count())}
        added_images: List[str] = []

        for raw in raw_paths:
            path = Path(raw)
            if not path.exists():
                continue

            if path.is_dir():
                folder_images = sorted(
                    p for p in path.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
                for img_path in folder_images:
                    img_str = str(img_path)
                    if img_str not in existing:
                        self.image_list.addItem(img_str)
                        existing.add(img_str)
                    added_images.append(img_str)
            elif path.suffix.lower() == ".json":
                self.calibration_path.setText(str(path))
                self._save_settings()
                self._log(f"Kalibrierungsdatei geladen: {path}")
            elif path.suffix.lower() in IMAGE_EXTENSIONS:
                img_str = str(path)
                if img_str not in existing:
                    self.image_list.addItem(img_str)
                    existing.add(img_str)
                added_images.append(img_str)

        if not added_images:
            return

        self.settings.setValue("undistort/last_image_dir", str(Path(added_images[0]).parent))
        self._select_list_path(added_images[0])

        if self.auto_run.isChecked():
            self._run_undistort_paths(added_images)
        else:
            self._preview_file(added_images[0])

    def _select_list_path(self, target_path: str) -> None:
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.text() == target_path:
                self.image_list.setCurrentItem(item)
                break

    def _add_images(self) -> None:
        start_dir = str(self.settings.value("undistort/last_image_dir", str(Path.cwd())))
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Eingabebilder auswählen",
            start_dir,
            IMAGE_FILE_FILTER,
        )
        if files:
            self.handle_paths(files)

    def _remove_selected(self) -> None:
        for item in self.image_list.selectedItems():
            row = self.image_list.row(item)
            self.image_list.takeItem(row)

    def _clear_images(self) -> None:
        self.image_list.clear()
        self.preview_widget.clear_pair()
        self.copy_result_button.setEnabled(False)
        self.adaptation_label.setText("Bereit – Bild ablegen oder auswählen.")

    def _browse_calibration(self) -> None:
        start_dir = str(self.settings.value("undistort/last_calibration_dir", str(PROJECT_ROOT)))
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Kalibrierungs-JSON auswählen",
            start_dir,
            "JSON Files (*.json)",
        )
        if file_path:
            self.calibration_path.setText(file_path)
            self.settings.setValue("undistort/last_calibration_dir", str(Path(file_path).parent))
            self._save_settings()
            self._refresh_current_preview()

    def _browse_output_dir(self) -> None:
        start_dir = self.output_dir.text().strip() or str(PROJECT_ROOT / "Undistorted")
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Ausgabeordner auswählen",
            start_dir,
        )
        if folder_path:
            self.output_dir.setText(folder_path)
            self._save_settings()

    def _open_output_dir(self) -> None:
        out_text = self.output_dir.text().strip()
        if not out_text:
            self._log("Ausgabeordner ist leer.")
            return
        out_path = Path(out_text).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_path)))

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _load_calibration(self) -> Optional[CalibrationData]:
        calibration_path = self.calibration_path.text().strip()
        if not calibration_path:
            self._log("Kalibrierungs-JSON ist leer.")
            return None
        try:
            return load_calibration_json(calibration_path)
        except Exception as exc:
            self._log(f"Kalibrierung konnte nicht geladen werden: {exc}")
            return None

    def _on_calibration_edited(self) -> None:
        self._save_settings()
        self._refresh_current_preview()

    def _on_option_toggled(self) -> None:
        self._save_settings()
        self._refresh_current_preview()

    def _on_selected_image_changed(self, current, _previous) -> None:
        if current is None:
            return
        self._preview_file(current.text())

    def _refresh_current_preview(self) -> None:
        current = self.image_list.currentItem()
        if current is not None:
            self._preview_file(current.text())
        elif self.preview_widget.before_image is not None:
            self._preview_numpy_image(self.preview_widget.before_image)

    def _preview_file(self, image_path: str) -> None:
        image = read_image(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return
        self._preview_numpy_image(image)

    def _preview_numpy_image(self, image: np.ndarray) -> None:
        calibration = self._load_calibration()
        if calibration is None:
            self.preview_widget.set_pair(image, None)
            self.copy_result_button.setEnabled(False)
            self.adaptation_label.setText("Keine gültige Kalibrierung geladen.")
            return

        height, width = image.shape[:2]
        _, _, info = adapt_calibration_for_size(calibration, width, height)
        undistorted = undistort_image(image, calibration, self.keep_border.isChecked())
        self.preview_widget.set_pair(image, undistorted)
        self.copy_result_button.setEnabled(True)
        self.adaptation_label.setText(f"Anpassung: {info}")

    def _paste_from_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()
        if mime is None:
            self._log("Zwischenablage ist leer.")
            return

        if mime.hasUrls():
            local_paths = [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]
            if local_paths:
                self.handle_paths(local_paths)
                return

        if mime.hasImage():
            qimage = clipboard.image()
            if not qimage.isNull():
                self._process_in_memory_image(qimage, source_label="clipboard")
                return

        self._log("Keine Bilddaten oder Dateien in der Zwischenablage gefunden.")

    def _process_in_memory_image(self, qimage: QImage, source_label: str) -> None:
        calibration = self._load_calibration()
        if calibration is None:
            return

        output_dir_text = self.output_dir.text().strip()
        if not output_dir_text:
            self._log("Ausgabeordner ist leer.")
            return

        image = qimage_to_cv(qimage)
        height, width = image.shape[:2]
        _, _, info = adapt_calibration_for_size(calibration, width, height)
        undistorted = undistort_image(image, calibration, self.keep_border.isChecked())

        self.preview_widget.set_pair(image, undistorted)
        self.copy_result_button.setEnabled(True)
        self.adaptation_label.setText(f"Anpassung: {info}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir_text) / f"{source_label}_{timestamp}_undistorted.png"
        if write_image(output_path, undistorted):
            self._log(f"Gespeichert ({info}): {output_path}")
        else:
            self._log(f"Fehler beim Speichern: {output_path}")
        self._save_settings()

    def _copy_result_to_clipboard(self) -> None:
        after = self.preview_widget.after_image
        if after is None:
            self._log("Kein entzerrtes Bild für die Zwischenablage vorhanden.")
            return
        QApplication.clipboard().setImage(cv_to_qimage(after))
        self._log("Entzerrtes Ergebnis in die Zwischenablage kopiert.")

    def _run_undistort(self) -> None:
        image_paths = [self.image_list.item(i).text() for i in range(self.image_list.count())]
        if not image_paths:
            self._log("Keine Bilder ausgewählt.")
            return
        self._run_undistort_paths(image_paths)

    def _run_undistort_paths(self, image_paths: List[str]) -> None:
        calibration = self._load_calibration()
        if calibration is None:
            return

        output_dir_text = self.output_dir.text().strip()
        if not output_dir_text:
            self._log("Ausgabeordner ist leer.")
            return

        output_dir = Path(output_dir_text)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"Entzerre {len(image_paths)} Bild(er)...")
        saved_count = 0
        first_before: Optional[np.ndarray] = None
        first_after: Optional[np.ndarray] = None
        first_info = ""

        for path_str in image_paths:
            path = Path(path_str)
            image = read_image(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                self._log(f"Übersprungen (nicht lesbar): {path}")
                continue

            height, width = image.shape[:2]
            _, _, info = adapt_calibration_for_size(calibration, width, height)
            undistorted = undistort_image(image, calibration, self.keep_border.isChecked())
            out_path = output_dir / f"{path.stem}_undistorted.png"
            if write_image(out_path, undistorted):
                saved_count += 1
                self._log(f"OK [{info}] -> {out_path.name}")
                if first_before is None:
                    first_before = image
                    first_after = undistorted
                    first_info = info
            else:
                self._log(f"Fehler beim Schreiben: {out_path}")

        self._log(f"Fertig: {saved_count} Bild(er) gespeichert in {output_dir}")
        if first_before is not None and first_after is not None:
            self.preview_widget.set_pair(first_before, first_after)
            self.copy_result_button.setEnabled(True)
            self.adaptation_label.setText(f"Anpassung: {first_info}")

        self._save_settings()

    def _load_settings(self) -> None:
        calibration_path = str(self.settings.value("undistort/calibration_path", ""))
        default_calib = PROJECT_ROOT / "calibration.json"
        if (not calibration_path or not Path(calibration_path).exists()) and default_calib.exists():
            calibration_path = str(default_calib)

        output_dir = str(self.settings.value("undistort/output_dir", ""))
        if not output_dir or not Path(output_dir).exists():
            output_dir = str(PROJECT_ROOT / "Undistorted")

        keep_border = self.settings.value("undistort/keep_border", True, type=bool)
        auto_run = self.settings.value("undistort/auto_run", True, type=bool)

        if calibration_path:
            self.calibration_path.setText(calibration_path)
        if output_dir:
            self.output_dir.setText(output_dir)

        self.keep_border.setChecked(bool(keep_border))
        self.auto_run.setChecked(bool(auto_run))

    def _save_settings(self) -> None:
        self.settings.setValue("undistort/calibration_path", self.calibration_path.text())
        self.settings.setValue("undistort/output_dir", self.output_dir.text())
        self.settings.setValue("undistort/keep_border", self.keep_border.isChecked())
        self.settings.setValue("undistort/auto_run", self.auto_run.isChecked())


class CalibrationTab(QWidget):
    calibration_saved = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self.settings = QSettings()

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.ExtendedSelection)

        add_button = QPushButton("Add images")
        add_button.clicked.connect(self._add_images)

        remove_button = QPushButton("Remove selected")
        remove_button.setProperty("secondary", True)
        remove_button.clicked.connect(self._remove_selected)

        clear_button = QPushButton("Clear")
        clear_button.setProperty("secondary", True)
        clear_button.clicked.connect(self.image_list.clear)

        image_buttons = QHBoxLayout()
        image_buttons.addWidget(add_button)
        image_buttons.addWidget(remove_button)
        image_buttons.addWidget(clear_button)

        images_group = QGroupBox("Calibration images")
        images_layout = QVBoxLayout()
        images_layout.addWidget(self.image_list)
        images_layout.addLayout(image_buttons)
        images_group.setLayout(images_layout)

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(3, 40)
        self.rows_spin.setValue(7)

        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(3, 40)
        self.cols_spin.setValue(10)

        self.square_spin = QDoubleSpinBox()
        self.square_spin.setRange(5.0, 100.0)
        self.square_spin.setValue(27.0)
        self.square_spin.setSuffix(" mm")

        self.marker_spin = QDoubleSpinBox()
        self.marker_spin.setRange(3.0, 95.0)
        self.marker_spin.setValue(20.0)
        self.marker_spin.setSuffix(" mm")

        self.dictionary_combo = QComboBox()
        self.dictionary_combo.addItems(
            [
                "DICT_4X4_50",
                "DICT_5X5_100",
                "DICT_6X6_250",
                "DICT_7X7_250",
                "DICT_6X6_1000",
            ]
        )
        self.dictionary_combo.setCurrentText("DICT_6X6_250")

        params_form = QFormLayout()
        params_form.addRow("Rows", self.rows_spin)
        params_form.addRow("Cols", self.cols_spin)
        params_form.addRow("Square size", self.square_spin)
        params_form.addRow("Marker size", self.marker_spin)
        params_form.addRow("Dictionary", self.dictionary_combo)

        params_group = QGroupBox("Pattern settings")
        params_group.setLayout(params_form)

        self.output_path = QLineEdit(str(PROJECT_ROOT / "calibration.json"))
        output_button = QPushButton("Browse")
        output_button.setProperty("secondary", True)
        output_button.clicked.connect(self._browse_output)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_path)
        output_row.addWidget(output_button)

        output_group = QGroupBox("Output calibration file")
        output_layout = QVBoxLayout()
        output_layout.addLayout(output_row)
        output_group.setLayout(output_layout)

        self.run_button = QPushButton("Run calibration")
        self.run_button.clicked.connect(self._run_calibration)

        left_layout = QVBoxLayout()
        left_layout.addWidget(images_group)
        left_layout.addWidget(params_group)
        left_layout.addWidget(output_group)
        left_layout.addWidget(self.run_button)
        left_layout.addStretch(1)

        self.preview_widget = BeforeAfterPreview(allow_single_toggle=False)
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.preview_widget)
        preview_group.setLayout(preview_layout)

        self.save_preview_button = QPushButton("Save undistorted preview")
        self.save_preview_button.setProperty("secondary", True)
        self.save_preview_button.setEnabled(False)
        self.save_preview_button.clicked.connect(self._save_preview)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log)
        log_group.setLayout(log_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(preview_group)
        right_layout.addWidget(self.save_preview_button)
        right_layout.addWidget(log_group)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 3)
        self.setLayout(main_layout)

        self._load_settings()

    def _add_images(self) -> None:
        start_dir = str(self.settings.value("calibration/last_image_dir", str(Path.cwd())))
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select calibration images",
            start_dir,
            IMAGE_FILE_FILTER,
        )
        for file_path in files:
            if file_path:
                self.image_list.addItem(file_path)

        if files:
            self.settings.setValue("calibration/last_image_dir", str(Path(files[0]).parent))

    def _remove_selected(self) -> None:
        for item in self.image_list.selectedItems():
            row = self.image_list.row(item)
            self.image_list.takeItem(row)

    def _browse_output(self) -> None:
        start_path = str(self.settings.value("calibration/output_path", self.output_path.text()))
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save calibration file",
            start_path or "calibration.json",
            "JSON Files (*.json)",
        )
        if file_path:
            self.output_path.setText(file_path)
            self._save_settings()

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _save_preview(self) -> None:
        after = self.preview_widget.after_image
        if after is None:
            self._log("No undistorted preview available.")
            return

        start_dir = str(self.settings.value("calibration/last_preview_dir", str(Path.cwd())))
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save undistorted preview",
            start_dir,
            "PNG Files (*.png)",
        )
        if not file_path:
            return

        if write_image(file_path, after):
            self.settings.setValue("calibration/last_preview_dir", str(Path(file_path).parent))
            self._log(f"Saved undistorted preview to: {file_path}")
        else:
            self._log(f"Failed to save preview to: {file_path}")

    def _run_calibration(self) -> None:
        image_paths = [self.image_list.item(i).text() for i in range(self.image_list.count())]
        if not image_paths:
            self._log("No images selected.")
            return

        output_path = self.output_path.text().strip()
        if not output_path:
            self._log("Output path is empty.")
            return

        pattern = PatternConfig(
            pattern_type="charuco",
            rows=self.rows_spin.value(),
            cols=self.cols_spin.value(),
            square_size_mm=self.square_spin.value(),
            marker_size_mm=self.marker_spin.value(),
            dictionary=self.dictionary_combo.currentText(),
        )

        self._log("Running calibration...")
        try:
            result = calibrate_charuco(image_paths, pattern)
        except Exception as exc:
            self._log(f"Calibration failed: {exc}")
            return

        save_calibration_json(output_path, result, pattern, image_paths)
        self.calibration_saved.emit(str(Path(output_path).resolve()))

        used_images = [item for item in result.per_image if item.used]
        errors = [item.rms_error for item in used_images if item.rms_error is not None]

        self._log(f"Saved calibration to: {output_path}")
        self._log(f"Overall RMS: {result.rms:.4f} px")
        self._log(f"Used images: {len(used_images)} / {len(result.per_image)}")

        if errors:
            self._log(
                f"Per-image RMS min/mean/max: {min(errors):.4f} / {mean(errors):.4f} / {max(errors):.4f}"
            )

        preview_path = image_paths[0]
        preview_image = read_image(preview_path, cv2.IMREAD_COLOR)
        if preview_image is None:
            self._log(f"Preview image could not be read: {preview_path}")
            return

        height, width = preview_image.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            result.camera_matrix,
            result.dist_coeffs,
            (width, height),
            1.0,
            (width, height),
        )

        undistorted = cv2.undistort(
            preview_image,
            result.camera_matrix,
            result.dist_coeffs,
            None,
            new_camera_matrix,
        )

        self.preview_widget.set_pair(preview_image, undistorted)
        self.save_preview_button.setEnabled(True)
        self._save_settings()

    def _load_settings(self) -> None:
        self.rows_spin.setValue(int(self.settings.value("calibration/rows", self.rows_spin.value())))
        self.cols_spin.setValue(int(self.settings.value("calibration/cols", self.cols_spin.value())))
        self.square_spin.setValue(
            float(self.settings.value("calibration/square_size", self.square_spin.value()))
        )
        self.marker_spin.setValue(
            float(self.settings.value("calibration/marker_size", self.marker_spin.value()))
        )

        dictionary = str(
            self.settings.value("calibration/dictionary", self.dictionary_combo.currentText())
        )
        dictionary_index = self.dictionary_combo.findText(dictionary)
        if dictionary_index >= 0:
            self.dictionary_combo.setCurrentIndex(dictionary_index)

        output_path = str(self.settings.value("calibration/output_path", self.output_path.text()))
        if output_path and Path(output_path).parent.exists():
            self.output_path.setText(output_path)

    def _save_settings(self) -> None:
        self.settings.setValue("calibration/rows", self.rows_spin.value())
        self.settings.setValue("calibration/cols", self.cols_spin.value())
        self.settings.setValue("calibration/square_size", self.square_spin.value())
        self.settings.setValue("calibration/marker_size", self.marker_spin.value())
        self.settings.setValue("calibration/dictionary", self.dictionary_combo.currentText())
        self.settings.setValue("calibration/output_path", self.output_path.text())
