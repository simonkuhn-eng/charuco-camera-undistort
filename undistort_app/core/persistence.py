from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from .calibration import CalibrationResult, ImageStats
from .patterns import PatternConfig


@dataclass
class CalibrationData:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    pattern: PatternConfig
    rms: float
    per_image: List[ImageStats]
    created_utc: str
    source_images: List[str]


def save_calibration_json(
    path: str | Path,
    result: CalibrationResult,
    pattern: PatternConfig,
    source_images: Iterable[str | Path],
) -> None:
    data = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pattern": {
            "type": pattern.pattern_type,
            "rows": pattern.rows,
            "cols": pattern.cols,
            "square_size_mm": pattern.square_size_mm,
            "marker_size_mm": pattern.marker_size_mm,
            "dictionary": pattern.dictionary,
        },
        "image_size": [result.image_size[0], result.image_size[1]],
        "camera_matrix": result.camera_matrix.tolist(),
        "dist_coeffs": result.dist_coeffs.reshape(-1).tolist(),
        "rms": result.rms,
        "per_image": [
            {
                "path": stats.path,
                "detected_corners": stats.detected_corners,
                "used": stats.used,
                "rms_error": stats.rms_error,
            }
            for stats in result.per_image
        ],
        "source_images": [str(path) for path in source_images],
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")



def load_calibration_json(path: str | Path) -> CalibrationData:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    pattern_data = data.get("pattern", {})
    pattern = PatternConfig(
        pattern_type=pattern_data.get("type", "charuco"),
        rows=int(pattern_data.get("rows", 0)),
        cols=int(pattern_data.get("cols", 0)),
        square_size_mm=float(pattern_data.get("square_size_mm", 0.0)),
        marker_size_mm=float(pattern_data.get("marker_size_mm", 0.0)),
        dictionary=pattern_data.get("dictionary", "DICT_6X6_250"),
    )

    per_image_data = data.get("per_image", [])
    per_image = [
        ImageStats(
            path=item.get("path", ""),
            detected_corners=int(item.get("detected_corners", 0)),
            used=bool(item.get("used", False)),
            rms_error=item.get("rms_error", None),
        )
        for item in per_image_data
    ]

    return CalibrationData(
        camera_matrix=np.array(data.get("camera_matrix", []), dtype=np.float64),
        dist_coeffs=np.array(data.get("dist_coeffs", []), dtype=np.float64),
        image_size=tuple(data.get("image_size", [0, 0])),
        pattern=pattern,
        rms=float(data.get("rms", 0.0)),
        per_image=per_image,
        created_utc=data.get("created_utc", ""),
        source_images=[str(path) for path in data.get("source_images", [])],
    )



