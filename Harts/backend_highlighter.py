#!/usr/bin/env python3
"""
Backend version of PDF Keyword Highlighter
- Highlighter: highlight keywords in a PDF (used by /api/highlight-pdf)
- Spec Splitter: split large specs by SECTION 23 XX YY, starting a new file when XX changes,
  and name each file using the title line below the header (used by /api/split-spec)
"""

from pathlib import Path
import sys
import re
import zipfile
from typing import List, Tuple, Dict, Optional

# ----------------------- PyMuPDF import guard -----------------------
try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import pymupdf as fitz
    except Exception:
        raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")

if not hasattr(fitz, "open"):
    raise ImportError("Wrong fitz package detected. Run: pip uninstall fitz && pip install pymupdf")

# ----------------------- Shared helpers -----------------------------
def hex_to_rgb01(hex_str: str) -> Tuple[float, float, float]:
    s = hex_str.strip().lstrip('#')
    if len(s) == 3:
        s = ''.join(ch*2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"Bad color hex: {hex_str}")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)

def normalize_word_token(w: str) -> str:
    # Preserve diameter symbol and x (size notation), digits, letters; normalize case
    keep = re.sub(r"[^0-9A-Za-zØøxX]+", "", w)
    return keep.casefold()

def tokenize_phrase(phrase: str, case_sensitive: bool) -> List[str]:
    toks = [t for t in re.split(r"\s+", phrase.strip()) if t]
    if not toks:
        return []
    if case_sensitive:
        return [re.sub(r"[^0-9A-Za-zØøxX]+", "", t) for t in toks]
    else:
        return [re.sub(r"[^0-9A-Za-zØøxX]+", "", t).casefold() for t in toks]

def group_words_by_line(words: List[Tuple]) -> Dict[Tuple[int, int], List[Tuple]]:
    lines: Dict[Tuple[int, int], List[Tuple]] = {}
    for w in words:
        x0, y0, x1, y1, text, block_no, line_no, word_no = w
        key = (block_no, line_no)
        lines.setdefault(key, []).append(w)
    for key in lines:
        lines[key].sort(key=lambda w: w[7])
    return lines

def find_matches_in_line(line_words: List[Tuple], kw_tokens_list: List[List[str]],
                         whole_word: bool, case_sensitive: bool) -> List[Tuple[int, int, int]]:
    norm_tokens = [normalize_word_token(w[4]) if not case_sensitive else re.sub(r"[^0-9A-Za-zØøxX]+", "", w[4])
                   for w in line_words]
    results = []
    for ki, ktoks in enumerate(kw_tokens_list):
        if not ktoks:
            continue
        n = len(ktoks)
        for i in range(0, len(norm_tokens) - n + 1):
            window = norm_tokens[i:i+n]
            if whole_word:
                if window == ktoks:
                    results.append((i, i+n, ki))
            else:
                if all(ktoks[j] in window[j] for j in range(n)):
                    results.append((i, i+n, ki))
    return results

def add_highlight_for_span(page: fitz.Page, line_words: List[Tuple], start: int, end: int,
                           color_rgb01: Tuple[float, float, float], opacity: float):
    rect = fitz.Rect(line_words[start][0], line_words[start][1], line_words[start][2], line_words[start][3])
    for w in line_words[start+1:end]:
        rect |= fitz.Rect(w[0], w[1], w[2], w[3])
    annot = page.add_highlight_annot(rect)
    annot.set_colors(stroke=color_rgb01)
    annot.set_opacity(opacity)
    annot.update()

def load_default_keywords() -> str:
    """Load keywords from plan_keywords.txt if it exists."""
    keywords_file = Path("plan_keywords.txt")
    if keywords_file.exists():
        try:
            content = keywords_file.read_text(encoding='utf-8')
            keywords = [line.strip() for line in content.splitlines() if line.strip()]
            return ', '.join(keywords)
        except Exception as e:
            print(f"Warning: Could not read plan_keywords.txt: {e}")
    return ""

# ----------------------- Highlighter (existing) ----------------------
def highlight_pdf_backend(pdf_path: Path, output_path: Path, keywords: str,
                          case_sensitive: bool = False, whole_word: bool = False,
                          color_hex: str = "#fff200", opacity: float = 0.35) -> dict:
    """
    Highlight comma-separated keywords in a PDF and save to output_path.
    Returns dict with success status, message, file_size, highlights.
    """
    try:
        # If no keywords provided, try default file
        if not keywords.strip():
            keywords = load_default_keywords()
            if not keywords:
                return {"success": False, "error": "No keywords provided and plan_keywords.txt not found"}

        keyword_list = [s.strip() for s in keywords.split(',') if s.strip()]
        if not keyword_list:
            return {"success": False, "error": "No valid keywords found"}

        color_rgb = hex_to_rgb01(color_hex)
        kw_tokens_list = [tokenize_phrase(kw, case_sensitive) for kw in keyword_list]

        doc = fitz.open(str(pdf_path))
        total_hits = 0

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            words = page.get_text("words")
            if not words:
                continue
            lines = group_words_by_line(words)
            for _key, line_words in lines.items():
                if not line_words:
                    continue
                matches = find_matches_in_line(line_words, kw_tokens_list, whole_word, case_sensitive)
                for (start, end, _ki) in matches:
                    try:
                        add_highlight_for_span(page, line_words, start, end, color_rgb, opacity)
                        total_hits += 1
                    except Exception as e:
                        print(f"Warning: Could not highlight on page {page_index + 1}: {e}")
                        continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path), deflate=True, clean=True, garbage=3)
        doc.close()

        if not output_path.exists():
            return {"success": False, "error": "Failed to save output file"}

        file_size = output_path.stat().st_size
        return {
            "success": True,
            "message": f"Successfully highlighted {total_hits} matches",
            "file_size": file_size,
            "highlights": total_hits
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ----------------------- Spec Splitter (new) -------------------------
# Detect headers like "SECTION 23 88 00" -> (division, second, third);
# new output starts whenever the "second" pair changes.
# The output file name uses the first meaningful line below the header.

def _slug_name(s: str) -> str:
    s = re.sub(r'\s+', ' ', (s or '')).strip()
    s = s.replace('/', '-').replace('\\', '-').replace(':', ' - ')
    s = re.sub(r'[^A-Za-z0-9\-_. ()]+', '', s)
    return s[:80] if len(s) > 80 else s

def _compile_hdr_regex(user_regex: Optional[str]):
    regs = []
    if user_regex:
        try:
            regs.append(re.compile(user_regex, re.IGNORECASE | re.MULTILINE))
        except re.error:
            # ignore bad user regex, fall back to default
            pass
    # Default spec header: "SECTION 23 88 00"
    regs.append(re.compile(r'^\s*SECTION\s+(\d{2})\s+(\d{2})\s+(\d{2})\b', re.IGNORECASE | re.MULTILINE))
    return regs

def _extract_header(text: str, regs) -> Optional[Tuple[str, str, str, int]]:
    for rx in regs:
        m = rx.search(text)
        if m and m.lastindex and m.lastindex >= 3:
            # Return (division, second, third, start_char_index_in_text)
            return (m.group(1), m.group(2), m.group(3), m.start())
    return None

def _extract_title_below(text: str, hdr_pos: Optional[int]) -> str:
    """
    Return the first non-empty, non-'PART'/'SECTION' line after the header line.
    """
    lines = text.splitlines()
    # try to map header char position to a line index
    start_idx = 0
    if hdr_pos is not None:
        acc = 0
        for i, line in enumerate(lines):
            acc += len(line) + 1  # + newline
            if acc >= hdr_pos:
                start_idx = i + 1
                break
    for i in range(start_idx, min(start_idx + 20, len(lines))):
        cand = lines[i].strip()
        if not cand:
            continue
        u = cand.upper()
        if u.startswith("PART "):    # skip part headings
            continue
        if u.startswith("SECTION "): # skip repeated section headings
            continue
        return cand
    return "Untitled"

def split_spec_by_mech_second_pair(pdf_path: Path,
                                   division: str = '23',
                                   section_regex: Optional[str] = None,
                                   work_dir: Optional[Path] = None) -> dict:
    """
    Split a large spec PDF into files whenever the *second* pair in 'SECTION {division} XX YY'
    changes. Each output is named with the title line immediately under the header.
    Returns: {"success": bool, "zip_path": str, "count": int} or {"success": False, "error": str}
    """
    try:
        work_dir = work_dir or pdf_path.parent
        regs = _compile_hdr_regex(section_regex)

        src = fitz.open(str(pdf_path))
        groups = []  # each: {'second': '88', 'start': p0, 'end': p1, 'title': 'REFRIGERANT PIPING'}
        cur = None

        for pno in range(len(src)):
            text = src.load_page(pno).get_text("text") or ""
            hdr = _extract_header(text, regs)
            if hdr:
                first, second, third, pos = hdr
                if first == division:
                    if (cur is None) or (second != cur['second']):
                        # close previous range
                        if cur is not None:
                            cur['end'] = pno - 1
                            if cur['end'] >= cur['start']:
                                groups.append(cur)
                        # start new range for this second pair
                        title = _extract_title_below(text, pos)
                        cur = {'second': second, 'start': pno, 'end': pno, 'title': title}
                    else:
                        # same second -> continue current range
                        pass

        # close last open group
        if cur is not None:
            cur['end'] = len(src) - 1
            if cur['end'] >= cur['start']:
                groups.append(cur)

        if not groups:
            src.close()
            return {"success": False, "error": f"No Division {division} section headers found."}

        # Build per-group PDFs and zip them
        out_files: List[Path] = []
        for g in groups:
            ndoc = fitz.open()
            ndoc.insert_pdf(src, from_page=g['start'], to_page=g['end'])
            fname = f"Sec {division}-{g['second']} — {_slug_name(g['title'])}.pdf"
            out_path = work_dir / fname
            ndoc.save(str(out_path), deflate=True, clean=True, garbage=3)
            ndoc.close()
            out_files.append(out_path)

        src.close()

        zip_path = work_dir / f"{pdf_path.stem}-div{division}-subsections.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in out_files:
                zf.write(p, arcname=p.name)

        return {"success": True, "zip_path": str(zip_path), "count": len(out_files)}

    except Exception as e:
        return {"success": False, "error": f"Split error: {e}"}

# ----------------------- Optional: CLI test --------------------------
if __name__ == "__main__":
    # Simple CLI for local testing
    # 1) Highlight: python backend_highlighter.py highlight input.pdf output.pdf "kw1, kw2"
    # 2) Split spec: python backend_highlighter.py split-spec input.pdf outdir 23
    if len(sys.argv) < 2:
        print("Usage:\n"
              "  python backend_highlighter.py highlight input.pdf output.pdf 'kw1, kw2'\n"
              "  python backend_highlighter.py split-spec input.pdf outdir 23")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == "highlight" and len(sys.argv) >= 5:
        input_file = Path(sys.argv[2])
        output_file = Path(sys.argv[3])
        keywords = sys.argv[4]
        print(highlight_pdf_backend(input_file, output_file, keywords))
    elif mode == "split-spec" and len(sys.argv) >= 5:
        input_file = Path(sys.argv[2])
        outdir = Path(sys.argv[3]); outdir.mkdir(parents=True, exist_ok=True)
        division = sys.argv[4]
        result = split_spec_by_mech_second_pair(input_file, division=division, work_dir=outdir)
        print(result)
    else:
        print("Bad arguments. See usage above.")
        sys.exit(1)
