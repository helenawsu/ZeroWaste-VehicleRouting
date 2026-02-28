#!/usr/bin/env python3
"""
Saturday Landfill Route – grid-based extraction using img2table.

Uses OpenCV to detect table grid lines, then runs OCR per cell so columns/rows
follow the actual table structure instead of raw word positions.

  - Top left: Date, Truck weight at scale (from top of first page via Tesseract)
  - Table: Location, Days, Compost toters, # of toters, Compost dumpsters,
    # of dumpsters, Saturday toters, Saturday dumpsters, Time, Notes
  - # of toters, # of dumpsters, Saturday toters, Saturday dumpsters = digits only.

Requires: Python 3.9–3.13 (img2table does not support 3.14 yet).
          pip install img2table opencv-python-headless pdf2image pytesseract Pillow
          brew install poppler tesseract
"""

import csv
import re
import sys
from pathlib import Path

try:
    from img2table.document import PDF
    from img2table.ocr import TesseractOCR
except ImportError as e:
    print("Missing dependency:", e)
    print("Run: pip install img2table opencv-python-headless")
    sys.exit(1)

try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
except ImportError:
    pass  # only needed for Date/weight from top of page

# Same template as pdf_table_to_csv.py
TABLE_HEADER = [
    "Location",
    "Days",
    "Compost toters",
    "# of toters",
    "Compost dumpsters",
    "# of dumpsters",
    "Saturday toters",
    "Saturday dumpsters",
    "Time",
    "Notes",
]
NUM_TABLE_COLS = len(TABLE_HEADER)
DIGIT_COLUMN_INDICES = (3, 5, 6, 7)


def cell_to_digits(s):
    """Extract digits only for numeric columns."""
    if not s or not isinstance(s, str):
        return ""
    s = str(s).strip()
    m = re.search(r"\d+", s)
    return m.group(0) if m else ""


def normalize_row_digit_columns(cells):
    """Force # of toters, # of dumpsters, Saturday toters, Saturday dumpsters to digits only."""
    if len(cells) <= max(DIGIT_COLUMN_INDICES):
        return cells
    out = list(cells)
    for i in DIGIT_COLUMN_INDICES:
        if i < len(out):
            out[i] = cell_to_digits(out[i])
    return out


def extract_date_and_weight_from_image(pil_image, crop_top_fraction=0.25):
    """Run OCR on top portion of page to get Date and Truck weight at scale."""
    try:
        w, h = pil_image.size
        top = pil_image.crop((0, 0, w, int(h * crop_top_fraction)))
        img_gray = top.convert("L")
        text = pytesseract.image_to_string(img_gray)
        date_val = ""
        weight_val = ""
        date_re = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})")
        for line in text.splitlines():
            line_lower = line.lower()
            if "date" in line_lower and not date_val:
                m = date_re.search(line)
                if m:
                    date_val = m.group(1)
            if ("weight" in line_lower or "scale" in line_lower) and not weight_val:
                m = re.search(r"(\d+\.?\d*)", line)
                if m:
                    weight_val = m.group(1)
        return date_val, weight_val
    except Exception:
        return "", ""


def is_header_row(cells):
    """True if this row looks like the table header."""
    if not cells:
        return False
    first = (cells[0] if cells else "").strip().upper()
    return first in ("LOCATION", "LOC", "DAYS", "TIME", "NOTES")


def looks_like_location(cell_text):
    """True if cell could be a location name."""
    if not cell_text or len(str(cell_text).strip()) < 2:
        return False
    if not re.search(r"[A-Za-z]{2,}", str(cell_text)):
        return False
    u = str(cell_text).upper().strip()
    if u in ("LOCATION", "TIME", "NOTES", "DAYS", "TRASH", "TOTERS", "DUMPSTERS", "COMPOST", "SATURDAY"):
        return False
    return True


def is_garbage_row(cells):
    if not cells:
        return True
    first = (cells[0] if cells else "").strip()
    if not looks_like_location(first):
        return True
    if len(" ".join(str(c) for c in cells).strip()) < 4:
        return True
    return False


def pdf_to_clean_csv(pdf_path, out_csv_path=None, first_page=None, last_page=None):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    out_csv_path = Path(out_csv_path or pdf_path.with_suffix(".csv"))

    # Optional: get Date and Truck weight from top of first page (Tesseract on cropped image)
    meta_date = ""
    meta_weight = ""
    try:
        pages_pil = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=1)
        if pages_pil:
            meta_date, meta_weight = extract_date_and_weight_from_image(pages_pil[0])
    except Exception:
        pass

    # img2table: detect grid and extract table per page (OCR per cell)
    pdf = PDF(str(pdf_path), pdf_text_extraction=False)
    ocr = TesseractOCR(lang="eng", psm=6)
    tables_by_page = pdf.extract_tables(
        ocr=ocr,
        min_confidence=40,
        borderless_tables=True,  # also detect tables with faint or no grid lines
    )

    all_table_rows = []
    for page_idx, tables in (tables_by_page or {}).items():
        if not tables:
            continue
        # Use first (or largest) table on page
        table = max(tables, key=lambda t: t.df.size if t.df is not None else 0)
        if table.df is None or table.df.empty:
            continue
        df = table.df
        ncols = len(df.columns)
        for _, row in df.iterrows():
            cells = [str(row.iloc[j]).strip() if j < ncols else "" for j in range(NUM_TABLE_COLS)]
            if is_header_row(cells):
                continue
            if is_garbage_row(cells):
                continue
            cells = normalize_row_digit_columns(cells)
            all_table_rows.append(cells)

    meta_row = ["Date", "Truck weight at scale"] + [""] * (NUM_TABLE_COLS - 2)
    meta_row[0] = meta_date or ""
    meta_row[1] = meta_weight or ""
    out_rows = [meta_row, TABLE_HEADER] + all_table_rows

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_csv_path} (img2table grid-based extraction)")
    return out_csv_path


def main():
    script_dir = Path(__file__).resolve().parent
    default_pdf = script_dir / "data" / "Route Sheet- Staff Assignmets 2-20-22 (1).pdf"
    pdf_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else default_pdf
    if not pdf_path.is_absolute():
        pdf_path = script_dir / pdf_path
    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else pdf_path.with_suffix(".csv")
    try:
        pdf_to_clean_csv(pdf_path, out_csv_path=out_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
