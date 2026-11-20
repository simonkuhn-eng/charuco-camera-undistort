from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .patterns import PatternConfig, create_charuco_board


@dataclass
class ImageStats:
    path: str
    detected_corners: int
    used: bool
    rms_error: Optional[float]


@dataclass
class CalibrationResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    rms: float
    per_image: List[ImageStats]


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image safely from a path that may contain non-ASCII characters."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _detect_charuco(
    gray: np.ndarray,
    board: cv2.aruco_CharucoBoard,
    dictionary: cv2.aruco_Dictionary,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if charuco_corners is None or charuco_ids is None or len(charuco_corners) == 0:
            return None, None, 0
        return charuco_corners, charuco_ids, len(charuco_corners)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) == 0:
        return None, None, 0

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners,
        ids,
        gray,
        board,
    )

    if retval is None or charuco_corners is None or charuco_ids is None:
        return None, None, 0

    return charuco_corners, charuco_ids, len(charuco_corners)


def _charuco_reprojection_error(
    board: cv2.aruco_CharucoBoard,
    corners: np.ndarray,
    ids: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Optional[float]:
    if not hasattr(board, "matchImagePoints"):
        return None

    obj_points, img_points = board.matchImagePoints(corners, ids)
    if obj_points is None or img_points is None or len(obj_points) == 0:
        return None

    projected, _ = cv2.projectPoints(
        obj_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    diff = img_points.reshape(-1, 2) - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def calibrate_charuco(
    image_paths: Iterable[str | Path],
    pattern: PatternConfig,
    min_corners: int = 20,
) -> CalibrationResult:
    board, dictionary = create_charuco_board(pattern)

    all_corners: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []
    per_image: List[ImageStats] = []
    used_indices: List[int] = []
    image_size: Optional[Tuple[int, int]] = None

    for path in image_paths:
        path = Path(path)
        image = read_image(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            per_image.append(ImageStats(str(path), 0, False, None))
            continue

        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size

        if image_size == current_size:
            charuco_corners, charuco_ids, corner_count = _detect_charuco(
                image,
                board,
                dictionary,
            )
        elif current_size == (image_size[1], image_size[0]):
            charuco_corners, charuco_ids, corner_count = None, None, 0
            for rotate_code in (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE):
                cand_corners, cand_ids, cand_count = _detect_charuco(
                    cv2.rotate(image, rotate_code),
                    board,
                    dictionary,
                )
                if cand_count > corner_count:
                    charuco_corners, charuco_ids, corner_count = cand_corners, cand_ids, cand_count
                if corner_count >= min_corners:
                    break
        else:
            raise ValueError(
                "Image size mismatch. Expected "
                f"{image_size[0]}x{image_size[1]}, got "
                f"{current_size[0]}x{current_size[1]} for {path}. "
                "This can happen if some images are cropped or from another resolution."
            )

        used = charuco_corners is not None and charuco_ids is not None and corner_count >= min_corners
        per_image.append(ImageStats(str(path), corner_count, used, None))

        if used:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            used_indices.append(len(per_image) - 1)

    if not all_corners:
        raise ValueError("No valid Charuco detections found. Check the board settings and images.")

    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_corners,
            all_ids,
            board,
            image_size,
            None,
            None,
        )
    else:
        all_obj_points: List[np.ndarray] = []
        all_img_points: List[np.ndarray] = []
        matched_corners: List[np.ndarray] = []
        matched_ids: List[np.ndarray] = []
        valid_used_indices: List[int] = []

        for idx, (corners, ids) in enumerate(zip(all_corners, all_ids)):
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            if obj_pts is not None and img_pts is not None and len(obj_pts) >= 4:
                all_obj_points.append(obj_pts)
                all_img_points.append(img_pts)
                matched_corners.append(corners)
                matched_ids.append(ids)
                valid_used_indices.append(used_indices[idx])
            else:
                per_image[used_indices[idx]].used = False

        if not all_obj_points:
            raise ValueError("No valid Charuco point matches found for calibration.")

        all_corners = matched_corners
        all_ids = matched_ids
        used_indices = valid_used_indices

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            all_obj_points,
            all_img_points,
            image_size,
            None,
            None,
        )

    for idx, (rvec, tvec, corners, ids) in enumerate(zip(rvecs, tvecs, all_corners, all_ids)):
        image_index = used_indices[idx]
        per_image[image_index].rms_error = _charuco_reprojection_error(
            board,
            corners,
            ids,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )

    if image_size is None:
        raise ValueError("No readable images found for calibration.")

    return CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        rms=float(rms),
        per_image=per_image,
    )
