#!/usr/bin/env python3
"""
Minimal script: detect table grid (no merging), then output
- All columns as images (column_0.png ... column_N.png)
- First 5 rows as one image (first_5_rows.png)
- Every row as an image (row_0.png ... row_M.png) for checking.

No OCR, no CSV/JSON.

Usage: python grid_extract_debug.py [path/to/route_sheet.pdf]
Default PDF: data/Route Sheet- Staff Assignmets 2-20-22 (1).pdf

Requires: opencv-python, pdf2image, pillow, numpy. Optional: poppler.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# --- Config ---
DPI = 300
NUM_TABLE_COLUMNS = 10
FIRST_N_ROWS = 5
OUTPUT_DIR = "grid_debug"

# --- Grid detection (no projection fallback); frac 0.40–0.45 finds grid without being too strict ---
H_KERNEL_FRAC = 0.15
H_KERNEL_MIN_LEN = 80
V_KERNEL_FRAC = 0.15
V_KERNEL_MIN_LEN = 80
ROW_MERGE_DIST = 8
COL_MERGE_DIST = 20
MIN_ROW_HEIGHT_PX = 18


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def deskew(img: np.ndarray) -> np.ndarray:
    """Correct scan tilt using minAreaRect on non-zero pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if coords.size < 100:
        return img
    pts = coords[:, [1, 0]].astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    if angle < -45:
        angle = angle + 90
    if abs(angle) < 0.3:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _line_centers_from_positions(
    positions: list[int],
    gap: int = 5,
) -> list[int]:
    if not positions:
        return []
    positions = sorted(set(positions))
    clusters = []
    cluster = [positions[0]]
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] <= gap:
            cluster.append(positions[i])
        else:
            clusters.append((min(cluster), max(cluster)))
            cluster = [positions[i]]
    if cluster:
        clusters.append((min(cluster), max(cluster)))
    return [(c[0] + c[1]) // 2 for c in clusters]


def _boundaries_from_centers(
    centers: list[int],
    dim_max: int,
) -> list[tuple[int, int]]:
    if not centers:
        return [(0, dim_max)]
    boundaries = [(0, centers[0])]
    for i in range(len(centers) - 1):
        boundaries.append((centers[i], centers[i + 1]))
    boundaries.append((centers[-1], dim_max))
    return boundaries


def _row_boundaries_from_projection(
    row_sums: np.ndarray,
    min_row_height: int = 12,
    max_row_height: int = 100,
) -> list[tuple[int, int]]:
    h = len(row_sums)
    kernel = np.ones(5) / 5
    smoothed = np.convolve(row_sums.astype(float), kernel, mode="same")
    valley_ys = []
    for y in range(2, h - 2):
        if smoothed[y] <= smoothed[y - 1] and smoothed[y] <= smoothed[y + 1]:
            if smoothed[y] < np.median(smoothed):
                valley_ys.append(y)
    valley_centers = _line_centers_from_positions(valley_ys, gap=ROW_MERGE_DIST)
    boundaries = []
    prev = 0
    for v in valley_centers:
        if v - prev >= min_row_height and v - prev <= max_row_height:
            boundaries.append((prev, v))
        prev = v
    if h - prev >= min_row_height:
        boundaries.append((prev, h))
    if not boundaries:
        return [(0, h)]
    return boundaries


def _merge_thin_rows(
    row_boundaries: list[tuple[int, int]],
    min_height: int,
) -> list[tuple[int, int]]:
    if not row_boundaries or len(row_boundaries) <= 1:
        return row_boundaries
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(row_boundaries):
        y0, y1 = row_boundaries[i]
        height = y1 - y0
        while height < min_height and i + 1 < len(row_boundaries):
            next_y0, next_y1 = row_boundaries[i + 1]
            y1 = next_y1
            height = y1 - y0
            i += 1
        out.append((y0, y1))
        i += 1
    return out


def detect_grid(img: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Returns (col_boundaries, row_boundaries). No column merging; raw detected columns and rows."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    # Rows: only horizontal lines spanning most of page width (full-width rule)
    h_kernel_len = max(int(w * H_KERNEL_FRAC), H_KERNEL_MIN_LEN)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    row_ys = [y for y in range(h) if np.any(h_lines[y, :] > 0)]
    row_centers = _line_centers_from_positions(row_ys, gap=ROW_MERGE_DIST)
    row_boundaries = _boundaries_from_centers(row_centers, dim_max=h)
    row_boundaries = _merge_thin_rows(row_boundaries, min_height=MIN_ROW_HEIGHT_PX)

    # Columns: only vertical lines spanning most of page height (full-height rule)
    v_kernel_len = max(int(h * V_KERNEL_FRAC), V_KERNEL_MIN_LEN)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    col_xs = [x for x in range(w) if np.any(v_lines[:, x] > 0)]
    col_centers = _line_centers_from_positions(col_xs, gap=COL_MERGE_DIST)
    if len(col_centers) < 3:
        col_boundaries = [(w * i // NUM_TABLE_COLUMNS, w * (i + 1) // NUM_TABLE_COLUMNS) for i in range(NUM_TABLE_COLUMNS)]
    else:
        col_boundaries = _boundaries_from_centers(col_centers, dim_max=w)
    # No merging: use raw detected columns (6th and 7th by index for output)

    return col_boundaries, row_boundaries


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_pdf = script_dir / "data" / "route_sheet.pdf"
    pdf_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else default_pdf
    if not pdf_path.is_absolute():
        pdf_path = script_dir / pdf_path

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = script_dir / OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)
    print(f"Output dir: {out_dir}", flush=True)

    print("Loading PDF page 1...", flush=True)
    pages = convert_from_path(str(pdf_path), dpi=DPI)
    if not pages:
        print("No pages in PDF.", file=sys.stderr)
        sys.exit(1)
    pil_img = pages[0]
    cv2_img = pil_to_cv2(pil_img)

    print("Deskewing...", flush=True)
    cv2_img = deskew(cv2_img)
    h, w = cv2_img.shape[:2]

    print("Detecting grid...", flush=True)
    col_boundaries, row_boundaries = detect_grid(cv2_img)
    n_cols = len(col_boundaries)
    n_rows = len(row_boundaries)

    print(f"  Rows:   {n_rows}", flush=True)
    print(f"  Columns: {n_cols}", flush=True)

    y0_all = row_boundaries[0][0]
    y1_all = row_boundaries[-1][1]

    # All columns: one image per column (full height)
    for i in range(n_cols):
        x0, x1 = col_boundaries[i]
        crop = cv2_img[y0_all:y1_all, max(0, x0):min(w, x1)]
        path = out_dir / f"column_{i}.png"
        cv2.imwrite(str(path), crop)
    print(f"  Saved: column_0.png ... column_{n_cols - 1}.png ({n_cols} columns)", flush=True)

    # First 5 rows: one image (all columns × first 5 rows)
    n_rows_to_show = min(FIRST_N_ROWS, len(row_boundaries))
    if row_boundaries and n_cols > 0 and n_rows_to_show > 0:
        row_images = []
        for r in range(n_rows_to_show):
            ry0, ry1 = row_boundaries[r]
            cells = [cv2_img[ry0:ry1, max(0, col_boundaries[i][0]):min(w, col_boundaries[i][1])] for i in range(n_cols)]
            row_images.append(np.hstack(cells))
        first_5_img = np.vstack(row_images)
        path = out_dir / "first_5_rows.png"
        cv2.imwrite(str(path), first_5_img)
        print(f"  Saved: {path.name} (first {n_rows_to_show} rows × {n_cols} columns)", flush=True)

    # Every row: one image per row (row_0.png ... row_{n_rows-1}.png)
    if row_boundaries and n_cols > 0:
        for r in range(len(row_boundaries)):
            ry0, ry1 = row_boundaries[r]
            cells = [cv2_img[ry0:ry1, max(0, col_boundaries[i][0]):min(w, col_boundaries[i][1])] for i in range(n_cols)]
            row_img = np.hstack(cells)
            path = out_dir / f"row_{r}.png"
            cv2.imwrite(str(path), row_img)
        print(f"  Saved: row_0.png ... row_{len(row_boundaries) - 1}.png ({len(row_boundaries)} rows)", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
