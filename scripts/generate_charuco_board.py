from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from undistort_app.core.patterns import PatternConfig, create_charuco_board


def _mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Charuco board for printing.")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--square-size-mm", type=float, required=True)
    parser.add_argument("--marker-size-mm", type=float, required=True)
    parser.add_argument("--dictionary", default="DICT_6X6_250")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    parser.add_argument("--output", default="charuco_board.png")
    args = parser.parse_args()

    pattern = PatternConfig(
        pattern_type="charuco",
        rows=args.rows,
        cols=args.cols,
        square_size_mm=args.square_size_mm,
        marker_size_mm=args.marker_size_mm,
        dictionary=args.dictionary,
    )

    board, _ = create_charuco_board(pattern)

    board_width_px = _mm_to_px(args.cols * args.square_size_mm, args.dpi)
    board_height_px = _mm_to_px(args.rows * args.square_size_mm, args.dpi)
    margin_px = _mm_to_px(args.margin_mm, args.dpi)

    if hasattr(board, "generateImage"):
        board_img = board.generateImage((board_width_px, board_height_px))
    else:
        board_img = board.draw((board_width_px, board_height_px))

    canvas = np.full(
        (board_height_px + margin_px * 2, board_width_px + margin_px * 2),
        255,
        dtype=np.uint8,
    )

    canvas[margin_px : margin_px + board_height_px, margin_px : margin_px + board_width_px] = board_img

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG for {output_path}")
    encoded.tofile(str(output_path))

    print(f"Saved: {output_path}")
    print("Print at 100% scale and disable any fit-to-page option.")


if __name__ == "__main__":
    main()

