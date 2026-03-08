import os
import re
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import easyocr

# TrOCR (handwritten) for cell OCR after grid extract
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False


# --- Config -----------------------------------------------------------------

# Columns: using 0‑based indices into detected column bands (must have exactly 10 columns)
TOTERS_COL_IDX = 6
DUMPSTERS_COL_IDX = 7

REQUIRED_COLS = 10   # table must have exactly 10 columns
REQUIRED_ROWS = 54   # total data rows across all Saturday Landfill pages (must match locations)

# Grid detection (from grid_extract_debug.py)
H_KERNEL_FRAC = 0.08 #affects rows
H_KERNEL_MIN_LEN = 80
V_KERNEL_FRAC = 0.10
V_KERNEL_MIN_LEN = 80
ROW_MERGE_DIST = 8
COL_MERGE_DIST = 20

CELL_CROP_PAD_PX = 2

HEADER_KEYWORDS = ["toter", "dumpster", "cust", "location"]

CELL_OCR_MIN_SIDE = 48
CELL_OCR_MIN_HEIGHT = 128  # upscale cell to at least this height before OCR

DEBUG_SAVE_GRID = True   # save row/col crops for review when debug_root set
DEBUG_SAVE_CELLS = True  # save each toters/dumpsters cell crop when debug_root set


@dataclass
class PageMeta:
    page_index: int
    title: Optional[str]
    date: Optional[str]
    truck_weight: Optional[str]


@dataclass
class GridifiedPage:
    """Result of Stage 1 (gridify): one Saturday Landfill page with validated grid. No cell OCR yet."""
    cv_img: np.ndarray
    row_bounds: List[int]
    col_bounds: List[int]
    first_data_row: int
    last_data_row: int
    page_index: int
    title: Optional[str]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _location_to_filename(location: str) -> str:
    """Sanitize location name for use in debug image filenames."""
    s = re.sub(r'[/\\:*?"<>|]', "_", location)
    s = re.sub(r"\s+", "_", s.strip())
    return s or "unknown"


def save_grid_debug_visuals(
    cv_img: np.ndarray,
    row_bounds: List[int],
    col_bounds: List[int],
    page_index: int,
    first_data_row: int,
    last_data_row: int,
    header_row: Optional[int],
    debug_root: str,
) -> None:
    """
    Save images for review: each row band, each column band, and an overview with grid drawn.
    Uses absolute path so output location is independent of cwd.
    """
    debug_root = os.path.abspath(debug_root)
    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1
    if n_rows <= 0 or n_cols <= 0:
        return
    page_dir = os.path.join(debug_root, f"page{page_index + 1}")
    ensure_dir(page_dir)
    rows_dir = os.path.join(page_dir, "rows")
    cols_dir = os.path.join(page_dir, "cols")
    ensure_dir(rows_dir)
    ensure_dir(cols_dir)

    # Overview: full page with grid lines drawn
    overview = cv_img.copy()
    if len(row_bounds) > 2:
        for y in row_bounds[1:-1]:
            cv2.line(overview, (0, y), (overview.shape[1], y), (0, 255, 0), 1)
    if len(col_bounds) > 2:
        for x in col_bounds[1:-1]:
            cv2.line(overview, (x, 0), (x, overview.shape[0]), (255, 0, 0), 1)
    if header_row is not None and 0 <= header_row < len(row_bounds):
        y_h = row_bounds[header_row]
        cv2.line(overview, (0, y_h), (overview.shape[1], y_h), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(page_dir, "grid_overview.png"), overview)

    # Each row band
    for r in range(n_rows):
        y1, y2 = row_bounds[r], row_bounds[r + 1]
        if y2 <= y1:
            continue
        row_crop = cv_img[y1:y2, :]
        label = "header" if r == header_row else ("data" if first_data_row <= r <= last_data_row else "skip")
        cv2.imwrite(os.path.join(rows_dir, f"row{r:02d}_{label}.png"), row_crop)

    # Each column band
    for c in range(n_cols):
        x1, x2 = col_bounds[c], col_bounds[c + 1]
        if x2 <= x1:
            continue
        col_crop = cv_img[:, x1:x2]
        cv2.imwrite(os.path.join(cols_dir, f"col{c:02d}.png"), col_crop)

    # Index file for quick reference
    with open(os.path.join(page_dir, "index.txt"), "w", encoding="utf-8") as f:
        f.write(f"page {page_index + 1}: {n_rows} rows, {n_cols} columns\n")
        f.write(f"header_row={header_row}, data rows {first_data_row}..{last_data_row} ({last_data_row - first_data_row + 1} data rows)\n")
        f.write("rows/ = one image per row (row00_skip.png, ...)\n")
        f.write("cols/ = one image per column (col00.png ...)\n")


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def deskew(cv_img: np.ndarray) -> np.ndarray:
    """Simple deskew based on minAreaRect of non‑white pixels."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thr)
    if coords is None:
        return cv_img
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    (h, w) = cv_img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(cv_img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# --- Grid detection (from grid_extract_debug.py) ---

def _line_centers_from_positions(positions: List[int], gap: int = 5) -> List[int]:
    """Cluster nearby positions into line centers; return center of each cluster."""
    if not positions:
        return []
    positions = sorted(set(positions))
    clusters: List[Tuple[int, int]] = []
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


def _boundaries_from_centers_tuples(
    centers: List[int],
    dim_max: int,
) -> List[Tuple[int, int]]:
    """Given line centers and extent, return list of (start, end) band boundaries."""
    if not centers:
        return [(0, dim_max)]
    centers = sorted(centers)
    boundaries: List[Tuple[int, int]] = [(0, centers[0])]
    for i in range(len(centers) - 1):
        boundaries.append((centers[i], centers[i + 1]))
    boundaries.append((centers[-1], dim_max))
    return boundaries


def _detect_grid(img: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Grid detection from grid_extract_debug.py.
    Returns (col_boundaries, row_boundaries) as list of (start, end) per band.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    # Rows: horizontal lines spanning most of page width
    h_kernel_len = max(int(w * H_KERNEL_FRAC), H_KERNEL_MIN_LEN)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    row_ys = [y for y in range(h) if np.any(h_lines[y, :] > 0)]
    row_centers = _line_centers_from_positions(row_ys, gap=ROW_MERGE_DIST)
    row_boundaries = _boundaries_from_centers_tuples(row_centers, dim_max=h)

    # Columns: vertical lines spanning most of page height
    v_kernel_len = max(int(h * V_KERNEL_FRAC), V_KERNEL_MIN_LEN)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    col_xs = [x for x in range(w) if np.any(v_lines[:, x] > 0)]
    col_centers = _line_centers_from_positions(col_xs, gap=COL_MERGE_DIST)
    if len(col_centers) < 3:
        col_boundaries = [
            (w * i // REQUIRED_COLS, w * (i + 1) // REQUIRED_COLS)
            for i in range(REQUIRED_COLS)
        ]
    else:
        col_boundaries = _boundaries_from_centers_tuples(col_centers, dim_max=w)

    return col_boundaries, row_boundaries


def detect_grid_lines(cv_img: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Detect table grid using logic from grid_extract_debug.py.
    First and last detected columns are treated as margins and dropped; only inner
    table columns are returned.
    Returns (row_bounds, col_bounds) as lists of pixel positions:
    row_bounds[i]..row_bounds[i+1] is row i; same for col_bounds.
    """
    col_boundaries, row_boundaries = _detect_grid(cv_img)
    # Convert list of (start, end) to list of ints [start0, end0, end1, ...] for compatibility
    row_bounds = [row_boundaries[0][0]] + [r[1] for r in row_boundaries]
    col_bounds = [col_boundaries[0][0]] + [c[1] for c in col_boundaries]
    # Drop first and last column (margins outside table)
    if len(col_bounds) > 2:
        col_bounds = col_bounds[1:-1]
    return row_bounds, col_bounds


def extract_header_title(cv_img: np.ndarray, reader: easyocr.Reader) -> Optional[str]:
    """OCR the top 15% of the page to get the route title."""
    h, w = cv_img.shape[:2]
    top = int(0.0 * h)
    bottom = int(0.15 * h)
    crop = cv_img[top:bottom, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray, detail=0, paragraph=True)
    if not result:
        return None
    text = " ".join(result)
    return text.strip() or None


def is_saturday_landfill_title(title: Optional[str]) -> bool:
    if not title:
        return False
    t = title.lower()
    return "saturday" in t and "landfill" in t


def _cell_to_grayscale_upscale(cell_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = 4.0
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_w = max(new_w, CELL_OCR_MIN_SIDE)
    new_h = max(new_h, CELL_OCR_MIN_HEIGHT)  # at least 128px height for better OCR
    gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Contrast stretching
    lo, hi = np.percentile(gray, (5, 95))
    if hi > lo:
        gray = np.clip((gray - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return gray


def _parse_digit_result(text: str) -> Optional[int]:
    text = text.strip()
    if not text:
        return None
    m = re.search(r"\d+", text)
    if not m:
        return None
    s = m.group(0)
    # Values on sheet are 0‑20 typically; if OCR gave "547" keep first digit.
    if len(s) > 2:
        s = s[0]
    try:
        return int(s)
    except ValueError:
        return None


def load_trocr_handwritten() -> Tuple[Any, Any]:
    """Load TrOCR processor and model for handwritten text (for cell OCR after grid extract)."""
    if not TROCR_AVAILABLE:
        raise ImportError("TrOCR not available. Install: pip install transformers torch")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model


def ocr_cell_digits_trocr(
    cell_bgr: np.ndarray,
    processor: Any,
    model: Any,
    debug_ocr_input_path: Optional[str] = None,
) -> Optional[int]:
    """OCR a single cell using TrOCR handwritten model (used after grid extract)."""
    gray = _cell_to_grayscale_upscale(cell_bgr)
    if debug_ocr_input_path:
        cv2.imwrite(debug_ocr_input_path, gray)
    pil_img = Image.fromarray(gray).convert("RGB")
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return _parse_digit_result(text)


def ocr_cell_digits_easyocr(
    cell_bgr: np.ndarray,
    reader: easyocr.Reader,
    debug_ocr_input_path: Optional[str] = None,
) -> Optional[int]:
    """OCR for digits using EasyOCR (used for header/title; cells use TrOCR when available)."""
    gray = _cell_to_grayscale_upscale(cell_bgr)
    if debug_ocr_input_path:
        cv2.imwrite(debug_ocr_input_path, gray)
    result = reader.readtext(gray, detail=0, allowlist="0123456789", paragraph=True)
    if not result:
        return None
    text = " ".join(result)
    return _parse_digit_result(text)


def find_header_row(
    cv_img: np.ndarray,
    row_bounds: List[int],
    col_bounds: List[int],
    reader: easyocr.Reader,
) -> int:
    """
    Scan rows looking for header keywords in table interior columns.
    Raises ValueError if nothing matches.
    """
    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1

    start_row = 2  # skip date + weight rows
    end_row = max(0, n_rows - 2)  # skip last margin row

    # Use inner band of columns [1, n_cols-1) to avoid outer margins
    left_col = 1
    right_col = max(left_col + 1, n_cols - 1)

    for r in range(start_row, end_row):
        y1, y2 = row_bounds[r], row_bounds[r + 1]
        x1, x2 = col_bounds[left_col], col_bounds[right_col]
        crop = cv_img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        result = reader.readtext(gray, detail=0, paragraph=True)
        if not result:
            continue
        text = " ".join(result).lower()
        if any(k in text for k in HEADER_KEYWORDS):
            return r

    raise ValueError("Header row not found in grid.")


def extract_page_table_cells(
    cv_img: np.ndarray,
    row_bounds: List[int],
    col_bounds: List[int],
    first_data_row: int,
    last_data_row: int,
    reader: easyocr.Reader,
    page_index: int,
    debug_root: Optional[str] = None,
    trocr_processor: Any = None,
    trocr_model: Any = None,
    page_location_names: Optional[List[str]] = None,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Extract Saturday Toters/Dumpsters digits for the given data row range.
    Returns list of (toters, dumpsters) tuples in order of rows.
    If page_location_names is provided, debug cell images are named by location (e.g. MOFFIT_LIBRARY_toters.png).
    """
    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1
    if n_cols <= max(TOTERS_COL_IDX, DUMPSTERS_COL_IDX):
        raise ValueError(f"Not enough columns: found {n_cols}")

    out: List[Tuple[Optional[int], Optional[int]]] = []

    cell_debug_dir = None
    if debug_root and DEBUG_SAVE_CELLS:
        cell_debug_dir = os.path.join(debug_root, f"page{page_index + 1}", "cells")
        ensure_dir(cell_debug_dir)

    num_data_rows = last_data_row - first_data_row + 1
    location_names = page_location_names if page_location_names and len(page_location_names) >= num_data_rows else None

    for r in range(first_data_row, last_data_row + 1):
        if r < 0 or r >= n_rows:
            continue

        local_idx = r - first_data_row
        debug_label = _location_to_filename(location_names[local_idx]) if location_names else f"row{r:02d}"

        y1 = row_bounds[r] + CELL_CROP_PAD_PX
        y2 = row_bounds[r + 1] - CELL_CROP_PAD_PX
        if y2 <= y1:
            out.append((None, None))
            continue

        # Toters
        tx1 = col_bounds[TOTERS_COL_IDX] + CELL_CROP_PAD_PX
        tx2 = col_bounds[TOTERS_COL_IDX + 1] - CELL_CROP_PAD_PX
        # Dumpsters
        dx1 = col_bounds[DUMPSTERS_COL_IDX] + CELL_CROP_PAD_PX
        dx2 = col_bounds[DUMPSTERS_COL_IDX + 1] - CELL_CROP_PAD_PX

        toters_val: Optional[int]
        dumpsters_val: Optional[int]

        if tx2 > tx1:
            # Copy so OCR gets an independent array (one cell per call)
            t_crop = cv_img[y1:y2, tx1:tx2].copy()
            ocr_input_path = (
                os.path.join(cell_debug_dir, f"{debug_label}_toters_ocr_input.png")
                if cell_debug_dir else None
            )
            if trocr_processor is not None and trocr_model is not None:
                toters_val = ocr_cell_digits_trocr(
                    t_crop, trocr_processor, trocr_model, debug_ocr_input_path=ocr_input_path
                )
            else:
                toters_val = ocr_cell_digits_easyocr(t_crop, reader, debug_ocr_input_path=ocr_input_path)
            if cell_debug_dir:
                cv2.imwrite(os.path.join(cell_debug_dir, f"{debug_label}_toters.png"), t_crop)
        else:
            toters_val = None

        if dx2 > dx1:
            d_crop = cv_img[y1:y2, dx1:dx2].copy()
            ocr_input_path = (
                os.path.join(cell_debug_dir, f"{debug_label}_dumpsters_ocr_input.png")
                if cell_debug_dir else None
            )
            if trocr_processor is not None and trocr_model is not None:
                dumpsters_val = ocr_cell_digits_trocr(
                    d_crop, trocr_processor, trocr_model, debug_ocr_input_path=ocr_input_path
                )
            else:
                dumpsters_val = ocr_cell_digits_easyocr(d_crop, reader, debug_ocr_input_path=ocr_input_path)
            if cell_debug_dir:
                cv2.imwrite(os.path.join(cell_debug_dir, f"{debug_label}_dumpsters.png"), d_crop)
        else:
            dumpsters_val = None

        out.append((toters_val, dumpsters_val))

    return out


def load_locations(loc_path: str) -> List[str]:
    with open(loc_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


# ---------------------------------------------------------------------------
# Stage 1: Gridify — detect grid, validate 10 cols and row count. No cell OCR.
# ---------------------------------------------------------------------------

def run_gridify_stage(
    pdf_path: str,
    locations_path: str,
    debug_root: Optional[str],
) -> Tuple[List[GridifiedPage], List[str]]:
    """
    Stage 1 only: load PDF, deskew, detect grid, require exactly 10 columns,
    find header/data row range, save debug visuals. Validate total data row
    count equals len(locations). Do NOT run any cell OCR.
    Returns (gridified_pages, locations). Raises if grid is wrong.
    """
    locations = load_locations(locations_path)
    expected_rows = len(locations)
    print(f"[Stage 1: Gridify] Loaded {len(locations)} locations (expect {expected_rows} data rows).")

    # Create debug folder immediately (absolute path so location is predictable)
    if debug_root:
        debug_root = os.path.abspath(debug_root)
        ensure_dir(debug_root)
        print(f"Debug images will be saved to: {debug_root}")

    # Minimal OCR only for page type and header row (part of gridify)
    reader = easyocr.Reader(["en"], gpu=False)

    pages = convert_from_path(pdf_path, dpi=300)
    print(f"PDF has {len(pages)} pages.")

    gridified: List[GridifiedPage] = []

    for idx, pil_page in enumerate(pages):
        print(f"\n=== Page {idx + 1} ===")
        cv_img = pil_to_cv2(pil_page)
        cv_img = deskew(cv_img)

        # Always save deskewed page when debug_root set (so we have debug output even if grid fails)
        if debug_root:
            page_dir = os.path.join(debug_root, f"page{idx + 1}")
            ensure_dir(page_dir)
            cv2.imwrite(os.path.join(page_dir, "page_deskewed.png"), cv_img)

        title = extract_header_title(cv_img, reader)
        print(f"Header title: {title!r}")

        # Run grid detection for every page so we can save debug for all
        try:
            row_bounds, col_bounds = detect_grid_lines(cv_img)
        except Exception as e:
            if is_saturday_landfill_title(title):
                raise RuntimeError(f"Page {idx + 1}: grid detection failed: {e}") from e
            print(f"Grid detection failed (skipped): {e}")
            continue

        n_rows = len(row_bounds) - 1
        n_cols = len(col_bounds) - 1
        print(f"Grid detected: {n_rows} rows, {n_cols} columns")

        # Compute header/data row range (needed for debug visuals and extraction)
        header_row_val: Optional[int] = None
        try:
            header_row_val = find_header_row(cv_img, row_bounds, col_bounds, reader)
            first_data_row = header_row_val + 1
            last_data_row = max(first_data_row, n_rows - 2)
            print(f"Header row: {header_row_val}, data rows: {first_data_row}..{last_data_row}")
        except ValueError:
            first_data_row = 0
            last_data_row = max(0, n_rows - 2)
            print(f"No header (continuation page): data rows {first_data_row}..{last_data_row}")

        # Always save debug visuals for any page where grid was detected
        if debug_root:
            try:
                save_grid_debug_visuals(
                    cv_img,
                    row_bounds,
                    col_bounds,
                    idx,
                    first_data_row,
                    last_data_row,
                    header_row_val,
                    debug_root,
                )
                print(f"  Debug saved to {debug_root}/page{idx + 1}/")
            except Exception as e:
                print(f"  Warning: could not save debug visuals: {e}")

        if not is_saturday_landfill_title(title):
            print("Skipping (not Saturday Landfill).")
            continue

        # Now enforce exactly 10 columns (no fallback)
        if n_cols != REQUIRED_COLS:
            msg = f"Page {idx + 1}: need exactly {REQUIRED_COLS} columns, got {n_cols}."
            if debug_root:
                msg += f" Debug images saved to {os.path.abspath(debug_root)} for review."
            raise ValueError(msg)
        if n_cols <= max(TOTERS_COL_IDX, DUMPSTERS_COL_IDX):
            raise ValueError(
                f"Page {idx + 1}: columns {TOTERS_COL_IDX} and {DUMPSTERS_COL_IDX} required; only {n_cols} columns."
            )

        num_data_rows = last_data_row - first_data_row + 1
        print(f"Page {idx + 1}: {num_data_rows} data rows.")
        gridified.append(
            GridifiedPage(
                cv_img=cv_img,
                row_bounds=row_bounds,
                col_bounds=col_bounds,
                first_data_row=first_data_row,
                last_data_row=last_data_row,
                page_index=idx,
                title=title,
            )
        )

    total_data_rows = sum(
        p.last_data_row - p.first_data_row + 1 for p in gridified
    )
    print(f"\n[Stage 1: Gridify] Total data rows across pages: {total_data_rows} (required: {expected_rows})")
    if total_data_rows != expected_rows:
        raise ValueError(
            f"Gridify failed: total data rows {total_data_rows} != {expected_rows} locations. "
            "Fix grid or data row range; review debug_saturday_columns/page*/rows/ and page*/cols/."
        )
    print("[Stage 1: Gridify] Done and correct. Proceeding to Stage 2 (OCR).")

    if debug_root:
        with open(os.path.join(debug_root, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"Expected total data rows: {expected_rows}\n")
            f.write(f"Expected columns per page: {REQUIRED_COLS}\n")
            f.write(f"Actual total data rows (gridify): {total_data_rows}\n")
            f.write("\nPer-page: see pageN/index.txt and pageN/rows/ and pageN/cols/.\n")

    return gridified, locations


# ---------------------------------------------------------------------------
# Stage 2: OCR — run cell OCR only after gridify is done and correct.
# ---------------------------------------------------------------------------

def run_ocr_stage(
    gridified_pages: List[GridifiedPage],
    locations: List[str],
    out_json_path: str,
    debug_root: Optional[str],
) -> None:
    """
    Stage 2 only: run OCR on each data cell (toters/dumpsters) for every
    gridified page. Call only after run_gridify_stage has succeeded.
    """
    expected_rows = len(locations)
    print(f"\n[Stage 2: OCR] Running cell OCR for {len(gridified_pages)} page(s), {expected_rows} rows required.")

    reader = easyocr.Reader(["en"], gpu=False)
    trocr_processor, trocr_model = None, None
    if TROCR_AVAILABLE:
        print("Loading TrOCR handwritten model...")
        trocr_processor, trocr_model = load_trocr_handwritten()
        print("TrOCR ready.")
    else:
        print("TrOCR not available; using EasyOCR for cells.")

    all_values: List[Tuple[Optional[int], Optional[int]]] = []
    loc_offset = 0
    for p in gridified_pages:
        num_page_rows = p.last_data_row - p.first_data_row + 1
        page_locations = locations[loc_offset : loc_offset + num_page_rows]
        loc_offset += num_page_rows
        page_vals = extract_page_table_cells(
            p.cv_img,
            p.row_bounds,
            p.col_bounds,
            p.first_data_row,
            p.last_data_row,
            reader,
            page_index=p.page_index,
            debug_root=debug_root,
            trocr_processor=trocr_processor,
            trocr_model=trocr_model,
            page_location_names=page_locations,
        )
        print(f"Page {p.page_index + 1}: extracted {len(page_vals)} data rows.")
        all_values.extend(page_vals)

    if len(all_values) != expected_rows:
        raise ValueError(
            f"OCR stage: got {len(all_values)} rows, need {expected_rows}. Gridify was correct; check OCR."
        )

    records = [
        {
            "location": loc,
            "saturday_toters": toters,
            "saturday_dumpsters": dumpsters,
        }
        for loc, (toters, dumpsters) in zip(locations, all_values)
    ]
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[Stage 2: OCR] Wrote {len(records)} records to {out_json_path}")


def run_pipeline(
    pdf_path: str,
    locations_path: str,
    out_json_path: str,
    debug_root: Optional[str] = "debug_saturday_columns",
) -> None:
    # Stage 1: Gridify — no cell OCR until this is done and correct
    gridified_pages, locations = run_gridify_stage(
        pdf_path=pdf_path,
        locations_path=locations_path,
        debug_root=debug_root,
    )
    # Stage 2: OCR — only after gridify succeeded
    run_ocr_stage(
        gridified_pages=gridified_pages,
        locations=locations,
        out_json_path=out_json_path,
        debug_root=debug_root,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Saturday Landfill toters/dumpsters from route PDF.")
    parser.add_argument("pdf_path", help="Input Saturday Landfill PDF path")
    parser.add_argument(
        "--locations",
        default=os.path.join(os.path.dirname(__file__), "locations.txt"),
        help="Path to locations.txt",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "saturday_landfill.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--debug-root",
        default=os.path.join(os.path.dirname(__file__), "debug_saturday_columns"),
        help="Root directory for debug images",
    )

    args = parser.parse_args()

    run_pipeline(
        pdf_path=args.pdf_path,
        locations_path=args.locations,
        out_json_path=args.out,
        debug_root=args.debug_root,
    )

