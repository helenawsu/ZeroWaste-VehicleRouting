"""
Saturday Landfill Route Parser
Uses Claude Vision API to extract handwritten data from scanned route forms.

Requirements:
    pip install anthropic pdf2image pillow pandas

Usage:
    python parse_landfill.py --input route.pdf --output output.csv
    python parse_landfill.py --input scans/ --output output.csv  # folder of images/pdfs
"""

import anthropic
import base64
import argparse
import json
import re
import sys
from pathlib import Path
from io import BytesIO

import pandas as pd
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

ROUTE_KEYWORD = "SATURDAY LANDFILL"
LOCATIONS_FILE = Path(__file__).parent / "locations.txt"
MODEL = "claude-haiku-4-5-20251001"   # cheap + accurate enough for this task

# ── Load locations ─────────────────────────────────────────────────────────────

def load_locations(path: Path) -> list[str]:
    if not path.exists():
        print(f"ERROR: locations file not found at {path}")
        sys.exit(1)
    locations = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(locations)} locations from {path.name}")
    return locations

# ── Image loading ──────────────────────────────────────────────────────────────

def load_images_from_source(source: str) -> list[Image.Image]:
    """Accept a PDF, a single image, or a folder of images/PDFs."""
    p = Path(source)
    images = []

    if p.is_dir():
        files = sorted(p.glob("*"))
        for f in files:
            images.extend(_file_to_images(f))
    else:
        images.extend(_file_to_images(p))

    if not images:
        print("ERROR: No images found in source.")
        sys.exit(1)
    print(f"Loaded {len(images)} page(s) total.")
    return images


def _file_to_images(path: Path) -> list[Image.Image]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pdf2image import convert_from_path
            return convert_from_path(str(path), dpi=200)
        except ImportError:
            print("ERROR: pdf2image not installed. Run: pip install pdf2image")
            sys.exit(1)
    elif suffix in (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"):
        return [Image.open(path).convert("RGB")]
    return []

# ── Claude helpers ─────────────────────────────────────────────────────────────

def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.standard_b64encode(buf.getvalue()).decode()


def ask_claude(client: anthropic.Anthropic, img: Image.Image, prompt: str) -> str:
    b64 = image_to_base64(img)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ]
        }]
    )
    return response.content[0].text.strip()

# ── Page filtering ─────────────────────────────────────────────────────────────

TITLE_PROMPT = """Look at the top center of this scanned form image.
What is the printed route title? (e.g. 'SATURDAY LANDFILL ROUTE', 'MONDAY RECYCLING ROUTE')
Reply with ONLY the title text, nothing else. If you cannot find a title, reply 'UNKNOWN'."""


def get_page_title(client: anthropic.Anthropic, img: Image.Image) -> str:
    # Crop top 15% to reduce tokens + noise
    w, h = img.size
    top = img.crop((0, 0, w, int(h * 0.15)))
    return ask_claude(client, top, TITLE_PROMPT).upper()


def find_landfill_pages(client: anthropic.Anthropic, images: list[Image.Image]) -> list[tuple[int, Image.Image]]:
    """Return (original_index, image) for Saturday Landfill pages."""
    matches = []
    for i, img in enumerate(images):
        title = get_page_title(client, img)
        print(f"  Page {i+1}: title = '{title}'")
        if ROUTE_KEYWORD in title:
            matches.append((i, img))

    if len(matches) == 0:
        print("ERROR: No Saturday Landfill pages found.")
        sys.exit(1)
    if len(matches) > 2:
        print(f"WARNING: Found {len(matches)} matching pages, expected 2. Using first two.")
        matches = matches[:2]
    if len(matches) < 2:
        print(f"WARNING: Only found {len(matches)} Saturday Landfill page(s), expected 2.")

    print(f"Found {len(matches)} Saturday Landfill page(s) at indices: {[i for i,_ in matches]}")
    return matches

# ── Header extraction ──────────────────────────────────────────────────────────

HEADER_PROMPT = """This is the top portion of a Saturday Landfill Route form.
Extract the following handwritten fields:
- date (next to the 'Date:' label)
- truck_weight (next to the 'Truck weight at scale:' label)

Return ONLY valid JSON like:
{"date": "2/21/26", "truck_weight": ""}

Use empty string "" if a field is blank or unreadable."""


def extract_header(client: anthropic.Anthropic, img: Image.Image) -> dict:
    w, h = img.size
    top = img.crop((0, 0, w, int(h * 0.15)))
    raw = ask_claude(client, top, HEADER_PROMPT)
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        print(f"  WARNING: Could not parse header JSON: {raw}")
        return {"date": "", "truck_weight": ""}

# ── Table extraction ───────────────────────────────────────────────────────────

def build_table_prompt(locations: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {loc}" for i, loc in enumerate(locations))
    return f"""This is a scanned Saturday Landfill Route form.
The table has a LOCATION column on the left (printed text) and two columns of handwritten numbers:
- SATURDAY TOTERS
- SATURDAY DUMPSTERS

Below is the exact ordered list of location names that appear as printed row labels in this table.
Use these as anchors to find each row — match the printed location name in the table to the name in this list,
then read the handwritten digit(s) in the SATURDAY TOTERS and SATURDAY DUMPSTERS columns for that row.

Locations (in order, top to bottom):
{numbered}

Rules:
- Values are typically single or double digit integers (1-20)
- Use null for any blank or empty cell
- Do NOT confuse 0 with a blank cell — only use null if the cell is truly empty
- Some locations may only appear on this page (the route spans two pages) — for locations not visible on this page, still include them with null for both values

Return ONLY a JSON array with exactly {len(locations)} objects in the same order as the list above:
[
  {{"location": "MOFFIT LIBRARY", "toters": null, "dumpsters": 1}},
  {{"location": "MOSES/STEVENS", "toters": 2, "dumpsters": null}},
  ...
]

Do not include any explanation. Only return the JSON array."""


def extract_table_rows(client: anthropic.Anthropic, img: Image.Image, locations: list[str]) -> list[dict]:
    prompt = build_table_prompt(locations)
    raw = ask_claude(client, img, prompt)
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        rows = json.loads(clean)
        if not isinstance(rows, list):
            raise ValueError("Response is not a list")
        # Validate location names match expected — warn on mismatch
        for i, (row, loc) in enumerate(zip(rows, locations)):
            if row.get("location", "").upper() != loc.upper():
                print(f"  WARNING row {i+1}: expected '{loc}' but got '{row.get('location')}'")
        return rows
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ERROR parsing table JSON: {e}\n  Raw response: {raw}")
        return []

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse Saturday Landfill route forms.")
    parser.add_argument("--input", required=True, help="PDF file, image file, or folder")
    parser.add_argument("--output", default="output.csv", help="Output CSV path")
    args = parser.parse_args()

    locations = load_locations(LOCATIONS_FILE)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    print("\n── Loading images ──")
    images = load_images_from_source(args.input)

    print("\n── Identifying Saturday Landfill pages ──")
    landfill_pages = find_landfill_pages(client, images)

    # Extract header from first matching page
    print("\n── Extracting header fields ──")
    _, first_img = landfill_pages[0]
    header = extract_header(client, first_img)
    print(f"  Date: {header.get('date')} | Truck weight: {header.get('truck_weight')}")

    # Split locations across pages
    # Page 1 gets the first N rows, page 2 gets the remainder
    total = len(locations)
    rows_per_page = [None, None]

    if len(landfill_pages) == 2:
        # Ask Claude how many rows are on page 1
        # Simpler: just split evenly and let the assert catch mismatches
        # Or: pass all locations to each page and let Claude figure out which rows are present
        # Best approach: extract each page independently, concatenate, then assert total
        print("\n── Extracting table rows (page 1) ──")
        rows_p1 = extract_table_rows(client, landfill_pages[0][1], locations)

        print("\n── Extracting table rows (page 2) ──")
        rows_p2 = extract_table_rows(client, landfill_pages[1][1], locations)

        # Each page returns the rows it has; Claude will return null for rows not on that page
        # Merge: prefer non-null values, page 1 takes precedence for duplicates
        all_rows = []
        for i in range(total):
            r1 = rows_p1[i] if i < len(rows_p1) else {}
            r2 = rows_p2[i] if i < len(rows_p2) else {}
            merged = {
                "toters": r1.get("toters") if r1.get("toters") is not None else r2.get("dumpsters"),
                "dumpsters": r1.get("dumpsters") if r1.get("dumpsters") is not None else r2.get("dumpsters"),
            }
            all_rows.append(merged)
    else:
        print("\n── Extracting table rows (single page) ──")
        all_rows = extract_table_rows(client, landfill_pages[0][1], locations)

    # Assert row count
    if len(all_rows) != total:
        print(f"WARNING: Expected {total} rows but got {len(all_rows)}. Output may be misaligned.")
    else:
        print(f"\n✓ Row count matches: {total} rows")

    # Build DataFrame
    records = []
    for i, loc in enumerate(locations):
        row = all_rows[i] if i < len(all_rows) else {}
        records.append({
            "date": header.get("date", ""),
            "truck_weight": header.get("truck_weight", ""),
            "route_title": "SATURDAY LANDFILL ROUTE",
            "location": loc,
            "saturday_toters": row.get("toters"),
            "saturday_dumpsters": row.get("dumpsters"),
        })

    df = pd.DataFrame(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df)} rows to {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()