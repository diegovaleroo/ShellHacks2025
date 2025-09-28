#!/usr/bin/env python3
"""
backend_spec_splitter.py
Split a large spec PDF by Division and the "second pair" of numbers in headers like:
  SECTION 23 88 00
We group all consecutive pages for the same Division + second pair (e.g., 23-88)
into one file, named with the title line beneath the header.
"""

from pathlib import Path
import re
import zipfile
from typing import List, Tuple, Dict, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import pymupdf as fitz
    except Exception:
        raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")

if not hasattr(fitz, "open"):
    raise ImportError("Wrong 'fitz' package found. Use: pip uninstall -y fitz && pip install pymupdf")


# ---------- Helpers ----------

def _sanitize_filename(name: str, max_len: int = 120) -> str:
    """Make a safe filename segment."""
    name = re.sub(r"[^\w\s\-\.\(\)&]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", " ", name, flags=re.UNICODE).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip("_ .-")
    return name or "Untitled"

def _extract_lines(page: fitz.Page) -> List[str]:
    """Return a list of text lines for a page, preserving order."""
    text = page.get_text("text") or ""
    # Keep original line breaks; strip trailing spaces
    return [ln.rstrip() for ln in text.splitlines()]

def _find_headers_in_page(lines: List[str], header_re: re.Pattern) -> List[Tuple[int, re.Match]]:
    """Find all header matches (line index, match object) in this page."""
    hits = []
    for i, ln in enumerate(lines):
        m = header_re.match(ln.strip())
        if m:
            hits.append((i, m))
    return hits

def _title_after(lines: List[str], start_idx: int) -> str:
    """
    Return the first non-empty line after start_idx that doesn't look like another header.
    This will be used as the section title for naming.
    """
    for j in range(start_idx + 1, min(start_idx + 8, len(lines))):
        cand = lines[j].strip()
        if not cand:
            continue
        if cand.upper().startswith("SECTION "):
            continue
        # Skip "END OF SECTION ..." boilerplate
        if cand.upper().startswith("END OF SECTION"):
            continue
        # Good candidate
        return cand
    return "Section"

def _save_range(doc: fitz.Document, start_page: int, end_page: int, out_path: Path) -> None:
    """Save pages [start_page..end_page] inclusive to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    newdoc = fitz.open()
    # Insert the range in one shot
    newdoc.insert_pdf(doc, from_page=start_page, to_page=end_page)
    newdoc.save(str(out_path), deflate=True, clean=True, garbage=3)
    newdoc.close()


# ---------- Public API ----------

def split_spec_by_section(
    pdf_path: Path,
    out_zip_path: Path,
    division: str = "23",
    section_regex: str = r"^SECTION\s+(\d{2})\s+(\d{2})\s+(\d{2})",
) -> Dict:
    """
    Scan the PDF for headers matching section_regex.
    - Group consecutive pages by target Division (group 1) AND the "second pair" (group 2).
    - When (group2) changes OR we hit a header from a different division, we close the current chunk.
    - Name each chunk using: "DIV-SEC2 Title.pdf" where Title is the line under the header.

    Returns:
      {"success": True, "count": N, "zip_path": str(out_zip_path)}
      or {"success": False, "error": "..."}
    """
    try:
        if not Path(pdf_path).exists():
            return {"success": False, "error": f"Input file not found: {pdf_path}"}

        header_re = re.compile(section_regex, re.IGNORECASE)
        target_div = division.zfill(2)

        doc = fitz.open(str(pdf_path))
        total_pages = doc.page_count

        # Collect ALL headers (any division) with their page index, parsed groups, and title line
        headers: List[Dict] = []
        for pno in range(total_pages):
            page = doc.load_page(pno)
            lines = _extract_lines(page)
            hits = _find_headers_in_page(lines, header_re)
            for idx, m in hits:
                g1, g2, g3 = m.group(1), m.group(2), m.group(3)
                title = _title_after(lines, idx)
                headers.append({
                    "page": pno,
                    "div": g1,
                    "sec2": g2,
                    "sec3": g3,
                    "title": title
                })

        if not headers:
            doc.close()
            return {"success": False, "error": "No headers matching the regex were found."}

        # Iterate through headers to build chunks for the requested division
        pieces: List[Tuple[int, int, str, str]] = []  # (start_page, end_page, sec2, title)
        current_start: Optional[int] = None
        current_sec2: Optional[str] = None
        current_title: str = "Section"

        def close_current(end_page: int):
            nonlocal current_start, current_sec2, current_title, pieces
            if current_start is not None and current_sec2 is not None and end_page >= current_start:
                pieces.append((current_start, end_page, current_sec2, current_title))
            current_start = None
            current_sec2 = None

        for i, h in enumerate(headers):
            is_target = (h["div"] == target_div)
            next_page_boundary = headers[i + 1]["page"] if i + 1 < len(headers) else total_pages  # next header page (or end)

            if is_target:
                # Start or continue a chunk depending on sec2 changes
                if current_start is None:
                    # Begin new chunk at this header page
                    current_start = h["page"]
                    current_sec2 = h["sec2"]
                    current_title = h["title"]
                else:
                    # If second pair changed, close previous chunk up to page before this header
                    if h["sec2"] != current_sec2:
                        close_current(h["page"] - 1)
                        current_start = h["page"]
                        current_sec2 = h["sec2"]
                        current_title = h["title"]
                # If next header is a different division, close at the page before next header
                if i + 1 < len(headers) and headers[i + 1]["div"] != target_div:
                    close_current(next_page_boundary - 1)

            else:
                # Non-target division header; if a chunk is open, close before this page
                if current_start is not None:
                    close_current(h["page"] - 1)

        # If we ended the loop with an open chunk, close it at end of document
        if current_start is not None:
            close_current(total_pages - 1)

        if not pieces:
            doc.close()
            return {"success": False, "error": f"No sections found for Division {target_div}."}

        # Write parts to temp folder and zip them
        out_dir = out_zip_path.with_suffix("")  # folder name without .zip
        out_dir.mkdir(parents=True, exist_ok=True)

        written_files: List[Path] = []
        for start, end, sec2, title in pieces:
            safe_title = _sanitize_filename(title)
            fname = f"{target_div}-{sec2} {safe_title}.pdf"
            out_pdf = out_dir / fname
            _save_range(doc, start, end, out_pdf)
            written_files.append(out_pdf)

        doc.close()

        # Zip them up
        out_zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(out_zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in written_files:
                zf.write(str(f), arcname=f.name)

        return {"success": True, "count": len(written_files), "zip_path": str(out_zip_path)}

    except Exception as e:
        return {"success": False, "error": str(e)}
