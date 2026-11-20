# ChArUco Camera Undistort

Desktop tool built with PySide6 and OpenCV for one-time ChArUco camera calibration and image undistortion.

Designed for preparing reference photos of physical parts before importing them as canvas backgrounds in CAD for 3D printing and reverse engineering.

## Features

- Generate printable ChArUco calibration boards with custom row/column counts, square sizes, and ArUco dictionaries (`scripts/generate_charuco_board.py`).
- Calibrate camera intrinsics and lens distortion coefficients from multiple board viewpoints and inspect per-image reprojection RMS errors.
- Save and reload calibration profiles (`calibration.json`, see `calibration.example.json` for schema).
- Undistort batches of photos, clipboard images, or drag-and-drop PNG/JPG files with configurable alpha framing.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Print `charuco_board.png` (or generate a custom board):

```bash
python scripts/generate_charuco_board.py --rows 7 --cols 10 --square-size-mm 27 --marker-size-mm 20 --output charuco_board.png
```

2. Run the desktop application:

```bash
python -m undistort_app
```

You can also pass image file paths directly on startup, or launch via `Start_Undistort_App.bat` / `Start_Undistort_App.vbs` on Windows once `.venv` is created.
