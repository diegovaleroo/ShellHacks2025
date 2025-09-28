# backend_spec_batch_analyzer.py
"""
Run many section PDFs in parallel and zip the analysis PDFs.

Requires:
  pip install google-genai reportlab
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile, zipfile, shutil

# Reuse single-file analyzer
from backend_spec_analyzer import analyze_spec_backend

def _worker_analyze_one(in_path: str, out_dir: str, prompt: Optional[str], model: str) -> Dict[str, Any]:
    try:
        ip = Path(in_path)
        od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
        op = od / f"{ip.stem}-analysis.pdf"
        res = analyze_spec_backend(
            pdf_path=ip,
            output_path=op,
            prompt=prompt,
            model=model,
            title=f"Spec Extraction Report — {ip.name}",
        )
        res.update({"input": str(ip), "output": str(op)})
        return res
    except Exception as e:
        return {"success": False, "error": str(e), "input": in_path, "output": ""}

def analyze_sections_dir(
    sections_dir: Path,
    out_dir: Path,
    prompt: Optional[str],
    model: str = "gemini-2.5-flash",
    max_workers: int = 3,
) -> Dict[str, Any]:
    sections_dir = Path(sections_dir)
    out_dir = Path(out_dir)
    pdfs = sorted(p for p in sections_dir.glob("*.pdf") if p.stat().st_size > 0)
    if not pdfs:
        return {"success": False, "error": f"No PDFs in {sections_dir}", "count": 0}

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        futures = [ex.submit(_worker_analyze_one, str(p), str(out_dir), prompt, model) for p in pdfs]
        for fut in as_completed(futures):
            results.append(fut.result())
    ok = sum(1 for r in results if r.get("success"))
    return {"success": True, "count": len(results), "ok": ok, "results": results, "out_dir": str(out_dir)}

def analyze_sections_zip(
    zip_path: Path,
    output_zip_path: Optional[Path] = None,
    prompt: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    max_workers: int = 3,
) -> Dict[str, Any]:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return {"success": False, "error": f"ZIP not found: {zip_path}"}

    work = Path(tempfile.mkdtemp(prefix="spec-batch-"))
    extracted = work / "sections"
    out_dir = work / "analyses"
    extracted.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(extracted))

    batch = analyze_sections_dir(extracted, out_dir, prompt, model, max_workers)
    if not batch.get("success"):
        return batch

    if output_zip_path is None:
        output_zip_path = zip_path.with_name(f"{zip_path.stem}-analysis.zip")
    output_zip_path = Path(output_zip_path)
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(output_zip_path), "w", compression=zipfile.ZIP_DEFLATED) as z:
        for pdf in sorted(out_dir.glob("*.pdf")):
            z.write(str(pdf), arcname=pdf.name)

    batch["output_zip"] = str(output_zip_path)
    return batch
