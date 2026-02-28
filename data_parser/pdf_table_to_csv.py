#!/usr/bin/env python3
"""
Saturday Landfill Route template:
  - Top left: Date, Truck weight at scale
  - Table: Location, Days, Compost toters, # of toters, Compost dumpsters,
    # of dumpsters, Saturday toters, Saturday dumpsters, Time, Notes
  - # of toters, # of dumpsters, Saturday toters, Saturday dumpsters = digits only.

Requires: pip install pdf2image pytesseract Pillow; brew install poppler tesseract

For better table structure (grid lines, columns/rows respected): use
  pdf_table_to_csv_img2table.py  (requires: pip install img2table opencv-python-headless)
"""

import csv
import re
import sys
from pathlib import Path

try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image, ImageEnhance
except ImportError as e:
    print("Missing dependency:", e)
    print("Run: .venv/bin/python pdf_table_to_csv.py  (use venv Python)")
    print("pip install pdf2image pytesseract Pillow")
    print("brew install poppler tesseract")
    sys.exit(1)


# --- Config ---
ROW_TOLERANCE = 10
DPI = 300
TESSERACT_PSM = 6
PAGE_WIDTH_PX = int(8.5 * DPI)

# Saturday Landfill Route template (exact headers)
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
# Columns that must contain only digits (0-based indices)
DIGIT_COLUMN_INDICES = (3, 5, 6, 7)  # # of toters, # of dumpsters, Saturday toters, Saturday dumpsters

# Tokens to drop from OCR (noise). Do not add digits 0-9 so numeric columns can capture them.
NOISE_TOKENS = frozenset({
    "|", "[", "]", "{", "}", "(", ")", ",", ".", "-", "_", "~", "=", "°", "®",
    "sat", "sar", "sat]", "sat}", "sat|", "sar|", "96", "96g", "96¢", "oe", "ee",
    "a", "e", "i", "o", "f", "s", "t", "l", "J", "I",
    "SOE", "SC", "SS", "PE", "RE", "RS", "RT", "TE", "SE", "ES", "ET", "TS",
    "Ee", "ea", "ae", "oo", "ii", "cf", "fs", "of", "fa", "fy", "ff",
})
# Pattern: token is noise if it's only symbols/short junk
def is_noise_token(t):
    t = (t or "").strip()
    if not t or len(t) <= 1:
        return True
    if t.upper() in NOISE_TOKENS:
        return True
    if re.match(r"^[\|\{\}\[\]\~\=\-\_\.\,\;\:\s]+$", t):
        return True
    if re.match(r"^[0-9]{1,2}$", t) and len(t) <= 2:  # keep larger numbers
        return False
    return False


def preprocess_image(pil_image):
    img = pil_image.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img


def get_words_with_boxes(pil_image):
    img = preprocess_image(pil_image)
    data = pytesseract.image_to_data(img, config=f"--psm {TESSERACT_PSM} --oem 3", output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        if not text or is_noise_token(text):
            continue
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        words.append((text, x, y + h // 2, w, h))
    return words


def words_to_rows(words):
    if not words:
        return []
    sorted_by_y = sorted(words, key=lambda t: (t[2], t[1]))
    rows = []
    current_row = [sorted_by_y[0]]
    current_y = sorted_by_y[0][2]
    for w in sorted_by_y[1:]:
        if abs(w[2] - current_y) <= ROW_TOLERANCE:
            current_row.append(w)
        else:
            rows.append(current_row)
            current_row = [w]
            current_y = w[2]
    if current_row:
        rows.append(current_row)
    return rows


def find_header_row_index(rows):
    markers = ("LOCATION", "TIME", "NOTES", "TOTERS", "DUMPSTERS", "TRASH", "COMPOST", "DAYS")
    for i, row_words in enumerate(rows):
        text = " ".join(t[0] for t in row_words).upper()
        if any(m in text for m in markers):
            return i
    return None


def column_boundaries_from_header(header_row_words):
    if not header_row_words:
        return [(0, PAGE_WIDTH_PX)]
    sorted_by_x = sorted(header_row_words, key=lambda t: t[1])
    boundaries = [0]
    for t in sorted_by_x:
        boundaries.append(t[1] + t[3] // 2)  # x + width/2
    boundaries.append(PAGE_WIDTH_PX)
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def assign_row_to_columns(row_words, column_boundaries):
    ncol = len(column_boundaries)
    cells = [""] * ncol
    for t in row_words:
        text, x, _, w, _ = t
        center_x = x + w // 2
        best_c = min(range(ncol), key=lambda c: abs(center_x - (column_boundaries[c][0] + column_boundaries[c][1]) // 2))
        if cells[best_c]:
            cells[best_c] += " " + text
        else:
            cells[best_c] = text
    return [clean_cell(s) for s in cells]


def clean_cell(s):
    """Remove extra spaces and trailing noise from a cell."""
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^[\|\s\,\;\:\.]+|[\|\s\,\;\:\.]+$", "", s).strip()
    return s


def cell_to_digits(s):
    """Extract digits only for numeric columns. Returns empty string if no digit found."""
    if not s:
        return ""
    # First integer in the string (handles "3", " 5 ", "2yd", "10")
    m = re.search(r"\d+", (s or "").strip())
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


def looks_like_location(cell_text):
    """True if cell could be a location name (has letters, not just symbols)."""
    if not cell_text or len(cell_text) < 2:
        return False
    # Must contain at least one letter and look like a name (not only digits/symbols)
    if not re.search(r"[A-Za-z]{2,}", cell_text):
        return False
    # Reject pure header keywords
    u = cell_text.upper().strip()
    if u in ("LOCATION", "TIME", "NOTES", "DAYS", "TRASH", "TOTERS", "DUMPSTERS", "COMPOST", "SATURDAY"):
        return False
    return True


def is_garbage_row(cells):
    """True if row is clearly not data (all cells noise or empty)."""
    if not cells:
        return True
    first = (cells[0] if cells else "").strip()
    # Header-like: all short or known headers
    if first.upper() in ("LOCATION", "TIME", "NOTES", "DAYS", "TRASH", "TOTERS", "DUMPSTERS", "COMPOST", "SATURDAY"):
        return True
    # No plausible location in first column
    if not looks_like_location(first):
        return True
    # Entire row is very short tokens
    full = " ".join(cells).strip()
    if len(full) < 4:
        return True
    return False


def extract_date_and_weight(rows, header_idx):
    """From rows above header, try to get Date and Truck weight at scale."""
    date_val = ""
    weight_val = ""
    date_re = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})")
    weight_re = re.compile(r"(\d+\.?\d*)")
    for i in range(header_idx if header_idx is not None else 0):
        line = " ".join(t[0] for t in sorted(rows[i], key=lambda x: x[1]))
        line_lower = line.lower()
        if "date" in line_lower and not date_val:
            m = date_re.search(line)
            if m:
                date_val = m.group(1)
        if "weight" in line_lower or "scale" in line_lower:
            m = weight_re.search(line)
            if m:
                weight_val = m.group(1)
    return date_val, weight_val


def extract_page(pil_image):
    words = get_words_with_boxes(pil_image)
    rows = words_to_rows(words)
    if not rows:
        return "", "", [], []

    header_idx = find_header_row_index(rows)
    if header_idx is None:
        header_idx = 0
    date_val, weight_val = extract_date_and_weight(rows, header_idx)

    col_bounds = column_boundaries_from_header(rows[header_idx])
    if not col_bounds:
        col_bounds = [(0, PAGE_WIDTH_PX)]

    # Build table with fixed number of columns; normalize digit columns
    table_rows = []
    for r in rows[header_idx:]:
        cells = assign_row_to_columns(r, col_bounds)
        if len(cells) > NUM_TABLE_COLS:
            cells = cells[:NUM_TABLE_COLS]
        elif len(cells) < NUM_TABLE_COLS:
            cells = cells + [""] * (NUM_TABLE_COLS - len(cells))
        if is_garbage_row(cells):
            continue
        cells = normalize_row_digit_columns(cells)
        table_rows.append(cells)
    return date_val, weight_val, table_rows, rows[:header_idx]


def pdf_to_clean_csv(pdf_path, out_csv_path=None, first_page=None, last_page=None):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    out_csv_path = Path(out_csv_path or pdf_path.with_suffix(".csv"))

    kwargs = {"dpi": DPI}
    if first_page is not None:
        kwargs["first_page"] = first_page
    if last_page is not None:
        kwargs["last_page"] = last_page
    pages = convert_from_path(str(pdf_path), **kwargs)

    # First row: Date, Truck weight at scale (from first page that has them)
    meta_date = ""
    meta_weight = ""
    all_table_rows = []

    for page_num, img in enumerate(pages, start=first_page or 1):
        date_val, weight_val, table_rows, _ = extract_page(img)
        if date_val:
            meta_date = date_val
        if weight_val:
            meta_weight = weight_val
        for row in table_rows:
            all_table_rows.append(row)

    # Build CSV: row0 = Date, Truck weight at scale (top left), row1 = header, rest = data (10 cols)
    out_rows = []
    meta_row = ["Date", "Truck weight at scale"] + [""] * (NUM_TABLE_COLS - 2)
    meta_row[0] = meta_date or ""
    meta_row[1] = meta_weight or ""
    out_rows.append(meta_row)
    out_rows.append(TABLE_HEADER)
    out_rows.extend(all_table_rows)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_csv_path}")
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
