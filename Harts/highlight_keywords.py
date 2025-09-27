#!/usr/bin/env python3
"""
PDF Keyword Highlighter for Plan Notes
/Users/magdalenasaldubehere/Documents/Harts/4227-Mechanical Drawings.pdf
Given a multi-page PDF (10–40+ pages), this script searches for user-provided
keywords and highlights each occurrence directly in the PDF.

✅ Works best on vector/searchable PDFs (embedded text)
🟡 For scanned PDFs, optionally pre-run OCR (see --ocr flag + notes below)

Usage examples:
  python highlight_keywords.py input.pdf --out output.pdf \
      --keywords "supply air, return air, TDF, 24x12, Ø18" --whole-word

  # Read keywords (one per line) from a text file
  python highlight_keywords.py input.pdf --keywords-file keywords.txt

  # Auto-OCR (requires 'ocrmypdf' installed) if the input has no text layer
  python highlight_keywords.py input.pdf --ocr --out highlighted.pdf \
      --keywords "insulate, liner, 2\" thickness"

Install:
  pip install pymupdf
  # Optional OCR tools (macOS Homebrew):
  #   brew install ocrmypdf tesseract ghostscript qpdf

Notes:
- Default search is case-insensitive and whole-word (toggle via flags).
- Multi-word phrases are supported as contiguous words on the same line.
- Rotated text (90° notes) is supported since we operate on page words.
- Scanned PDFs: Use --ocr to call 'ocrmypdf --skip-text' and add a text layer.

Limitations (typical MVP trade-offs):
- Matches spanning line breaks are not combined.
- Hyphenated line ends may require OCR de-hyphenation if scanned.
"""

from __future__ import annotations
import argparse
import sys
import subprocess
from pathlib import Path
import re
import os
import webbrowser
from typing import List, Tuple, Dict

# Optional GUI file picker (for selecting the PDF without typing a path)
try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None

try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import pymupdf as fitz  # fallback import name in newer versions
    except Exception:
        print("PyMuPDF is required. Install with: python -m pip install --upgrade pymupdf", file=sys.stderr)
        raise

# Guard against wrong 'fitz' package (there is a different one)
if not hasattr(fitz, "open"):
    print(
        "Detected a different 'fitz' package. Fix by running:\n"
        "  pip uninstall -y fitz\n"
        "  python -m pip install --upgrade pymupdf",
        file=sys.stderr,
    )
    sys.exit(1)

# ----------------------------- Helpers ---------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Highlight keywords in a PDF of plan notes.")
    # Make the PDF positional argument OPTIONAL; if omitted we will prompt the user.
    p.add_argument("pdf", nargs="?", type=Path, help="Input PDF path")
    p.add_argument("--out", type=Path, default=None, help="Output PDF path (default: <input>-highlighted.pdf)")
    gk = p.add_mutually_exclusive_group(required=False)
    gk.add_argument("--keywords", type=str, default=None,
                    help="Comma-separated keywords / phrases to highlight")
    gk.add_argument("--keywords-file", type=Path, default=None,
                    help="Text file with one keyword/phrase per line")
    p.add_argument("--case-sensitive", action="store_true", help="Make search case-sensitive (default: off)")
    p.add_argument("--whole-word", action="store_true", help="Match whole words only (default: off)")
    p.add_argument("--ocr", action="store_true", help="If no text layer is detected, try OCR with 'ocrmypdf'")
    p.add_argument("--color", type=str, default="#fff200", help="Highlight color (hex, e.g., #fff200 for yellow)")
    p.add_argument("--opacity", type=float, default=0.35, help="Highlight opacity 0..1 (default: 0.35)")
    p.add_argument("--max-hits-per-page", type=int, default=5000, help="Safety limit to avoid runaway highlights")
    p.add_argument("--no-open", action="store_true", help="Do not auto-open the output PDF when done")
    return p.parse_args()


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


def load_keywords(args: argparse.Namespace) -> List[str]:
    if args.keywords_file:
        text = args.keywords_file.read_text(encoding='utf-8')
        kws = [ln.strip() for ln in text.splitlines() if ln.strip()]
    elif args.keywords:
        kws = [s.strip() for s in args.keywords.split(',') if s.strip()]
    else:
        print("No keywords provided. Use --keywords or --keywords-file", file=sys.stderr)
        sys.exit(2)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for kw in kws:
        key = kw if args.case_sensitive else kw.casefold()
        if key not in seen:
            seen.add(key)
            out.append(kw)
    return out


def has_text_layer(pdf_path: Path, sample_pages: int = 3) -> bool:
    """Check if PDF has a text layer by sampling the first few pages."""
    try:
        doc = fitz.open(str(pdf_path))  # Convert Path to string
        n = min(sample_pages, len(doc))
        
        for i in range(n):
            page = doc.load_page(i)
            words = page.get_text("words")
            if words:  # If any words found, it has text
                doc.close()
                return True
        
        doc.close()
        return False
    except Exception as e:
        print(f"Warning: Could not check text layer: {e}")
        return True  # Assume it has text if we can't check


def run_ocrmypdf(in_path: Path, out_path: Path) -> bool:
    """Run OCRmyPDF with --skip-text so it only OCRs pages without text.
    Returns True if succeeded, False otherwise.
    """
    cmd = [
        "ocrmypdf", "--skip-text", "--force-ocr", "--output-type", "pdf",
        str(in_path), str(out_path)
    ]
    try:
        print("[OCR] Running:", ' '.join(cmd))
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if r.returncode == 0:
            return True
        else:
            print("[OCR] ocrmypdf failed (non-zero exit).", file=sys.stderr)
            print(r.stderr.decode(errors='ignore'), file=sys.stderr)
            return False
    except FileNotFoundError:
        print("[OCR] 'ocrmypdf' not found. Install it or run OCR yourself.", file=sys.stderr)
        return False


# --------------------------- Utilities ----------------------------------

def pick_pdf_gui() -> Path | None:
    """Open a file picker to choose a PDF. Returns a Path or None if canceled/unavailable."""
    try:
        if tk is None:
            return None
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select plan PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        try:
            root.destroy()
        except Exception:
            pass
        if not path:
            return None
        return Path(path)
    except Exception:
        return None


def auto_open(path: Path) -> None:
    """Best-effort open of a file on macOS, Windows, or Linux. Falls back to webbrowser."""
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    
    try:
        if sys.platform.startswith("darwin"):  # macOS
            result = subprocess.run(["/usr/bin/open", str(path)], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to open with 'open': {result.stderr}")
        elif sys.platform.startswith("win"):  # Windows
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:  # Linux and others
            result = subprocess.run(["xdg-open", str(path)], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to open with 'xdg-open': {result.stderr}")
    except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
        # Fallback to webbrowser
        try:
            webbrowser.open(path.resolve().as_uri())
        except Exception as e2:
            raise RuntimeError(f"All open methods failed. OS error: {e}, Browser error: {e2}")


# --------------------------- Core logic ---------------------------------

def normalize_word_token(w: str) -> str:
    """Normalize a PDF word for matching: lowercase (casefold) and strip punctuation boundaries.
    Keep Ø/ø and digits & letters; drop most other non-alphanumerics.
    """
    # Preserve diameter symbol and x (size notation), numbers, letters
    keep = re.sub(r"[^0-9A-Za-zØøxX]+", "", w)
    return keep.casefold()


def tokenize_phrase(phrase: str, case_sensitive: bool) -> List[str]:
    # Split on whitespace; normalize tokens similar to words
    toks = [t for t in re.split(r"\s+", phrase.strip()) if t]
    if not toks:
        return []
    if case_sensitive:
        return [re.sub(r"[^0-9A-Za-zØøxX]+", "", t) for t in toks]
    else:
        return [re.sub(r"[^0-9A-Za-zØøxX]+", "", t).casefold() for t in toks]


def group_words_by_line(words: List[Tuple]) -> Dict[Tuple[int, int], List[Tuple]]:
    """Group PyMuPDF 'words' by (block_no, line_no), sorted by word number."""
    lines: Dict[Tuple[int, int], List[Tuple]] = {}
    for w in words:
        x0, y0, x1, y1, text, block_no, line_no, word_no = w
        key = (block_no, line_no)
        lines.setdefault(key, []).append(w)
    # Sort each line by word number
    for key in lines:
        lines[key].sort(key=lambda w: w[7])
    return lines


def find_matches_in_line(line_words: List[Tuple], kw_tokens_list: List[List[str]], whole_word: bool,
                         case_sensitive: bool) -> List[Tuple[int, int, int]]:
    """Return list of (start_index, end_index_exclusive, kw_index) matches on this line.
    Matching is contiguous word tokens on the same line.
    """
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
                # partial: require each keyword token to be a substring of corresponding word token
                if all(ktoks[j] in window[j] for j in range(n)):
                    results.append((i, i+n, ki))
    return results


def add_highlight_for_span(page: fitz.Page, line_words: List[Tuple], start: int, end: int,
                           color_rgb01: Tuple[float, float, float], opacity: float):
    rect = fitz.Rect(line_words[start][0], line_words[start][1], line_words[start][2], line_words[start][3])
    for w in line_words[start+1:end]:
        rect |= fitz.Rect(w[0], w[1], w[2], w[3])
    annot = page.add_highlight_annot(rect)
    # Only stroke color is used by highlight annotations; setting fill prints warnings.
    annot.set_colors(stroke=color_rgb01)
    annot.set_opacity(opacity)
    annot.update()


def highlight_keywords(pdf_in: Path, pdf_out: Path, keywords: List[str], case_sensitive: bool,
                        whole_word: bool, color_hex: str, opacity: float, hit_limit: int):
    try:
        color_rgb = hex_to_rgb01(color_hex)
    except ValueError as e:
        print(f"❌ Invalid color: {e}")
        return

    # Pre-tokenize keywords for efficient matching
    kw_tokens_list = [tokenize_phrase(kw, case_sensitive) for kw in keywords]

    try:
        doc = fitz.open(str(pdf_in))  # Ensure string path
    except Exception as e:
        print(f"❌ Error opening PDF: {e}")
        return

    total_hits = 0
    for page_index in range(doc.page_count):
        try:
            page = doc.load_page(page_index)
            words = page.get_text("words")  # list of (x0, y0, x1, y1, text, block, line, word)
            if not words:
                continue
            lines = group_words_by_line(words)
            for _key, line_words in lines.items():
                matches = find_matches_in_line(line_words, kw_tokens_list, whole_word, case_sensitive)
                for (start, end, _ki) in matches:
                    add_highlight_for_span(page, line_words, start, end, color_rgb, opacity)
                    total_hits += 1
                    if total_hits >= hit_limit:
                        print(f"Hit limit reached ({hit_limit}). Stopping further highlights.")
                        break
                if total_hits >= hit_limit:
                    break
            if total_hits >= hit_limit:
                break
        except Exception as e:
            print(f"⚠️  Error processing page {page_index + 1}: {e}")
            continue

    # Save the document
    try:
        # Ensure the output directory exists
        pdf_out.parent.mkdir(parents=True, exist_ok=True)
        
        save_args = {
            "deflate": True,
            "clean": True,
            "garbage": 3,
            "incremental": False,
        }
        doc.save(str(pdf_out), **save_args)  # Ensure string path
        doc.close()
        
        # Verify the file was actually saved
        if pdf_out.exists():
            file_size = pdf_out.stat().st_size
            print(f"✅ Saved highlighted PDF: {pdf_out} ({file_size} bytes, {total_hits} highlights)")
            return True
        else:
            print(f"❌ Error: Output file was not created at {pdf_out}")
            return False
            
    except Exception as e:
        print(f"❌ Error saving PDF: {e}")
        doc.close()
        return False


# ----------------------------- Main -------------------------------------

def main():
    print("=" * 60)
    print("         PDF KEYWORD HIGHLIGHTER")
    print("=" * 60)
    print()

    # ⬅️ pull in CLI flags (color, whole-word, --out, --no-open, --ocr, etc.)
    args = parse_args()

    # Always prompt for PDF file in a user-friendly way
    pdf_path = None
    while not pdf_path:
        try:
            user_input = input("📄 Enter the PDF filename (you can drag & drop the file here): \n> ").strip().strip('"')
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            sys.exit(0)

        if not user_input:
            print("❌ Please enter a filename.\n")
            continue

        pdf_path = Path(user_input)
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            print("💡 Make sure the file exists and try again.\n")
            pdf_path = None
            continue

        if pdf_path.suffix.lower() != ".pdf":
            print(f"❌ Please select a PDF file (got: {pdf_path.suffix})\n")
            pdf_path = None
            continue

    print(f"✅ Found PDF: {pdf_path.name}\n")

    # Get keywords in a user-friendly way
    keywords = None
    default_kw = Path("plan_keywords.txt")

    if default_kw.exists():
        use_default = input("📝 Found 'plan_keywords.txt'. Use it? (y/n, default=y): ").strip().lower()
        if use_default in ["", "y", "yes"]:
            try:
                text = default_kw.read_text(encoding="utf-8")
                keywords = [ln.strip() for ln in text.splitlines() if ln.strip()]
                print(f"✅ Loaded {len(keywords)} keywords from file")
            except Exception as e:
                print(f"❌ Error reading keywords file: {e}")

    if not keywords:
        while not keywords:
            try:
                kw_input = input("📝 Enter keywords to highlight (comma-separated): \n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye! 👋")
                sys.exit(0)

            if not kw_input:
                print("❌ Please enter at least one keyword.\n")
                continue

            keywords = [s.strip() for s in kw_input.split(",") if s.strip()]
            if not keywords:
                print("❌ No valid keywords found.\n")
                continue

    print(f"🔍 Will search for: {', '.join(keywords)}\n")

    # Generate output filename (respect --out if provided)
    output_path = Path(args.out) if args.out else pdf_path.with_name(pdf_path.stem + "-highlighted.pdf")

    # Check if output already exists
    if output_path.exists():
        overwrite = input(f"⚠️  Output file '{output_path.name}' already exists. Overwrite? (y/n, default=y): ").strip().lower()
        if overwrite not in ["", "y", "yes"]:
            # Generate unique name
            counter = 1
            while output_path.exists():
                output_path = pdf_path.with_name(f"{pdf_path.stem}-highlighted-{counter}.pdf")
                counter += 1

    # Check for text layer and (optionally) OCR if requested
    pdf_to_process = pdf_path
    if not has_text_layer(pdf_path):
        print("⚠️  This PDF does not appear to have a text layer.")
        if args.ocr:
            ocr_out = pdf_path.with_name(pdf_path.stem + ".ocr.pdf")
            if run_ocrmypdf(pdf_path, ocr_out):
                pdf_to_process = ocr_out
                print(f"✅ OCR complete: {ocr_out.name}")
            else:
                print("⚠️  OCR failed or not available. Proceeding without OCR.")
        else:
            print("💡 Re-run with --ocr to attempt automatic text recognition.\n")

    print("🔄 Processing PDF...")

    # 🔗 Call the highlighter and then auto-open on success
    ok = highlight_keywords(
        pdf_in=pdf_to_process,
        pdf_out=output_path,
        keywords=keywords,
        case_sensitive=args.case_sensitive,
        whole_word=args.whole_word,
        color_hex=args.color,
        opacity=args.opacity,
        hit_limit=args.max_hits_per_page,
    )

    if ok:
        print(f"📁 Output: {output_path}")
        if not args.no_open:
            try:
                auto_open(output_path)
                print("🚀 Opened highlighted PDF.")
            except Exception as e:
                print(f"⚠️  Could not auto-open the file: {e}")
                print("   You can open it manually from the path above.")
    else:
        print("❌ Highlighting failed; nothing to open.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
