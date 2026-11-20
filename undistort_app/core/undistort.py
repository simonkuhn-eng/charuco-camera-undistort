from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .calibration import read_image
from .persistence import CalibrationData


def write_image(path: str | Path, image: np.ndarray) -> bool:
    """Write an image safely to a path that may contain non-ASCII characters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def adapt_calibration_for_size(
    calibration: CalibrationData,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Scale and orient calibration intrinsics to match a target image resolution."""
    orig_w, orig_h = calibration.image_size
    K = calibration.camera_matrix.astype(np.float64).copy()
    dist = calibration.dist_coeffs.astype(np.float64).reshape(-1).copy()

    if orig_w <= 0 or orig_h <= 0:
        return K, dist, "unskaliert (keine Kalibrierungsgröße)"

    cal_w, cal_h = orig_w, orig_h
    is_rotated = (width != height) and (cal_w != cal_h) and ((width < height) != (cal_w < cal_h))

    if is_rotated:
        cal_w, cal_h = cal_h, cal_w
        K[0, 0], K[1, 1] = K[1, 1], K[0, 0]
        K[0, 2], K[1, 2] = K[1, 2], K[0, 2]
        if dist.size >= 4:
            dist[2], dist[3] = dist[3], dist[2]

    sx = width / cal_w
    sy = height / cal_h

    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy

    if not is_rotated and width == orig_w and height == orig_h:
        info = f"{orig_w}x{orig_h} (1:1)"
    else:
        details: List[str] = []
        if is_rotated:
            details.append("Hochformat gedreht" if width < height else "Querformat gedreht")
        if abs(sx - sy) < 0.01:
            details.append(f"Skalierung {sx:.3f}x")
        else:
            details.append(f"Skalierung {sx:.3f}x/{sy:.3f}x")
        info = f"{orig_w}x{orig_h} -> {width}x{height} ({', '.join(details)})"

    return K, dist, info


def _optimal_camera_matrix(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    keep_border: bool,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    alpha = 1.0 if keep_border else 0.0
    return cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        alpha,
        image_size,
    )


def undistort_image(
    image: np.ndarray,
    calibration: CalibrationData,
    keep_border: bool,
) -> np.ndarray:
    height, width = image.shape[:2]
    camera_matrix, dist_coeffs, _ = adapt_calibration_for_size(calibration, width, height)
    new_camera_matrix, roi = _optimal_camera_matrix(
        camera_matrix,
        dist_coeffs,
        (width, height),
        keep_border,
    )

    undistorted = cv2.undistort(
        image,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
    )

    if not keep_border:
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y : y + h, x : x + w]

    return undistorted


def undistort_files(
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    calibration: CalibrationData,
    keep_border: bool,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []

    for path in input_paths:
        path = Path(path)
        image = read_image(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue

        undistorted = undistort_image(image, calibration, keep_border)
        output_path = output_dir / f"{path.stem}_undistorted.png"
        if write_image(output_path, undistorted):
            outputs.append(output_path)

    return outputs
