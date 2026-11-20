from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import cv2

PatternType = Literal["charuco"]


@dataclass(frozen=True)
class PatternConfig:
    pattern_type: PatternType
    rows: int
    cols: int
    square_size_mm: float
    marker_size_mm: float
    dictionary: str


def get_aruco_dictionary(dictionary_name: str) -> cv2.aruco_Dictionary:
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    dictionary_id = getattr(cv2.aruco, dictionary_name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def create_charuco_board(config: PatternConfig) -> Tuple[cv2.aruco_CharucoBoard, cv2.aruco_Dictionary]:
    if config.pattern_type != "charuco":
        raise ValueError(f"Unsupported pattern type: {config.pattern_type}")

    dictionary = get_aruco_dictionary(config.dictionary)

    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (config.cols, config.rows),
            config.square_size_mm,
            config.marker_size_mm,
            dictionary,
        )
    else:
        board = cv2.aruco.CharucoBoard_create(
            config.cols,
            config.rows,
            config.square_size_mm,
            config.marker_size_mm,
            dictionary,
        )

    return board, dictionary
