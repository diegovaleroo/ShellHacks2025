#!/usr/bin/env python3
from __future__ import annotations

from flask import Flask, request, jsonify, send_file, render_template_string, url_for
import tempfile
import os
from pathlib import Path
import sys
import uuid
import threading
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# Make local backends importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Backends you provide alongside this file ---
try:
    from backend_highlighter import highlight_pdf_backend
except ImportError:
    print("ERROR: Could not import backend_highlighter.py")
    sys.exit(1)

try:
    from backend_spec_splitter import split_spec_by_section
except ImportError:
    print("ERROR: Could not import backend_spec_splitter.py")
    sys.exit(1)

try:
    from backend_spec_analyzer import analyze_spec_backend  # used by worker
except ImportError:
    print("ERROR: Could not import backend_spec_analyzer.py")
    sys.exit(1)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

# ----- Fixed configs (no user override) -----
FIXED_MODEL = "gemini-2.0-flash-lite"
MAX_WORKERS = 3

# ---------------- HTML loader ----------------
def get_html_content():
    try:
        with open('highlighterweb.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return (
            "<h1>Error: highlighterweb.html not found</h1>"
            "<p>Make sure highlighterweb.html is in the same directory as app.py</p>"
        )

# ---------------- Simple in-memory job registry ----------------
JOBS: dict[str, dict] = {}  # job_id -> {status, message, split_zip, analysis_zip, stats}

def _list_sections_dir_from_zip(zip_path: Path) -> Path:
    return Path(zip_path).with_suffix("")

# --------- Helpers to make a minimal PDF (for error reports) ----------
def _write_error_pdf(out_path: Path, title: str, message: str):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    flow = [
        Paragraph(title, styles["Heading1"]),
        Spacer(1, 12),
        Paragraph(message.replace("\n", "<br/>"), styles["Normal"]),
    ]
    doc.build(flow)

# --------- TOP-LEVEL WORKER (picklable) FOR MULTIPROCESSING ----------
def worker_analyze_one(in_path: str, out_dir: str, prompt: str | None, model: str) -> tuple[str, bool, str, str]:
    ip = Path(in_path)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    op = od / f"{ip.stem}-analysis.pdf"

    try:
        res = analyze_spec_backend(
            pdf_path=ip,
            output_path=op,
            prompt=prompt,
            model=model,
            title=f"Spec Extraction Report — {ip.name}",
        )
        ok = bool(res.get("success")) and op.exists()
        if not ok:
            _write_error_pdf(op, f"Analysis Error — {ip.name}", res.get("error", "Unknown analysis error"))
            return (ip.name, False, res.get("error", "Unknown analysis error"), str(op))
        return (ip.name, True, "", str(op))
    except Exception as e:
        try:
            _write_error_pdf(op, f"Analysis Error — {ip.name}", str(e))
            return (ip.name, False, str(e), str(op))
        except Exception as ee:
            err_txt = od / f"{ip.stem}-analysis-ERROR.txt"
            err_txt.write_text(f"Error: {e}\nSecondary error writing PDF: {ee}", encoding="utf-8")
            return (ip.name, False, f"{e} (and PDF write failed)", str(err_txt))

def _analyze_dir_to_zip(sections_dir: Path, analysis_zip_path: Path,
                        prompt: str | None, model: str, max_workers: int) -> dict:
    out_dir = Path(analysis_zip_path).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in Path(sections_dir).glob("*.pdf") if p.stat().st_size > 0)
    if not pdfs:
        raise RuntimeError(f"No PDFs found to analyze in {sections_dir}")

    results: list[tuple[str, bool, str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        futures = [ex.submit(worker_analyze_one, str(p), str(out_dir), prompt, model) for p in pdfs]
        for fut in as_completed(futures):
            results.append(fut.result())

    ok = sum(1 for _, success, _, _ in results if success)
    fail = len(results) - ok

    manifest = out_dir / "manifest.txt"
    lines = [
        f"Analysis manifest",
        f"Total: {len(results)}",
        f"Success: {ok}",
        f"Failed: {fail}",
        "",
    ]
    for name, success, err, output_path in sorted(results, key=lambda r: r[0]):
        status = "OK" if success else "FAILED"
        lines.append(f"{name}: {status}")
        if not success and err:
            lines.append(f"  Error: {err}")
        lines.append(f"  Output: {Path(output_path).name}")
    manifest.write_text("\n".join(lines), encoding="utf-8")

    import zipfile
    with zipfile.ZipFile(str(analysis_zip_path), "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out_dir.glob("*")):
            z.write(str(f), arcname=f.name)

    return {"total": len(results), "ok": ok, "fail": fail, "manifest": str(manifest), "out_dir": str(out_dir)}

# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template_string(get_html_content())

@app.route('/api/get-default-keywords')
def get_default_keywords():
    keywords_file = Path("plan_keywords.txt")
    if keywords_file.exists():
        try:
            content = keywords_file.read_text(encoding='utf-8')
            keywords = [line.strip() for line in content.splitlines() if line.strip()]
            return jsonify({'success': True, 'keywords': ', '.join(keywords), 'count': len(keywords)})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Could not read keywords file: {e}'})
    else:
        return jsonify({'success': False, 'error': 'plan_keywords.txt not found'})

@app.route('/api/highlight-pdf', methods=['POST'])
def highlight_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file uploaded'}), 400
        pdf_file = request.files['pdf']
        keywords = (request.form.get('keywords') or '').strip()

        if not pdf_file.filename:
            return jsonify({'error': 'No file selected'}), 400
        if not keywords:
            return jsonify({'error': 'No keywords provided'}), 400

        temp_dir = tempfile.mkdtemp()
        input_path = Path(temp_dir) / pdf_file.filename
        pdf_file.save(str(input_path))

        output_path = input_path.with_name(input_path.stem + '-highlighted.pdf')
        result = highlight_pdf_backend(
            pdf_path=input_path,
            output_path=output_path,
            keywords=keywords,
            case_sensitive=False,
            whole_word=False,
            color_hex="#fff200",
            opacity=0.35
        )

        if result.get("success") and output_path.exists():
            return send_file(
                str(output_path),
                as_attachment=True,
                download_name=f"{input_path.stem}-highlighted.pdf",
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': result.get("error", "Unknown error occurred")}), 500

    except Exception as e:
        print(f"[highlight_pdf] Error: {e}")
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/api/split-spec', methods=['POST'])
def split_spec():
    try:
        pdf_file = request.files.get('pdf')
        if not pdf_file or not pdf_file.filename:
            return jsonify({'error': 'No spec PDF provided'}), 400

        division = (request.form.get('division', '23') or '23').strip()
        if not division.isdigit() or len(division) != 2:
            return jsonify({'error': "Division must be 2 digits (e.g., '23')"}), 400

        section_regex = r'^\s*SECTION\s+(\d{2})\s+(\d{2})\s+(\d{2})(?:\s*[-–—]?\s*(.+))?\s*$'

        temp_dir = tempfile.mkdtemp()
        input_path = Path(temp_dir) / pdf_file.filename
        pdf_file.save(str(input_path))

        zip_path = input_path.with_name(f"{input_path.stem}-div{division}-subsections.zip")

        result = split_spec_by_section(
            pdf_path=input_path,
            out_zip_path=zip_path,
            division=division,
            section_regex=section_regex
        )

        if result.get("success") and zip_path.exists():
            return send_file(
                str(zip_path),
                as_attachment=True,
                download_name=zip_path.name,
                mimetype='application/zip'
            )

        return jsonify({'error': result.get('error', 'Unknown error while splitting spec')}), 500

    except Exception as e:
        print(f"[split_spec] Error: {e}")
        return jsonify({'error': f'Server error: {e}'}), 500

# -------- Async Split -> Analyze pipeline (fixed to MAX_WORKERS & FIXED_MODEL) --------
@app.route('/api/split-and-analyze-async', methods=['POST'])
def split_and_analyze_async():
    """
    Upload a spec PDF → split to sections (ZIP) immediately,
    and kick off background analysis for each section in parallel.
    Model fixed to gemini-2.0-flash-lite; workers fixed to 3.
    """
    try:
        pdf_file = request.files.get('pdf')
        if not pdf_file or not pdf_file.filename:
            return jsonify({'error': 'No spec PDF provided'}), 400

        division = (request.form.get('division') or '23').strip()
        if not division.isdigit() or len(division) != 2:
            return jsonify({'error': "Division must be 2 digits (e.g., '23')"}), 400

        section_regex = r'^\s*SECTION\s+(\d{2})\s+(\d{2})\s+(\d{2})(?:\s*[-–—]?\s*(.+))?\s*$'
        prompt = (request.form.get('prompt') or '').strip() or None

        job_id = str(uuid.uuid4())
        temp_dir = Path(tempfile.mkdtemp(prefix=f"job-{job_id}-"))
        in_pdf = temp_dir / pdf_file.filename
        pdf_file.save(str(in_pdf))

        zip_path = temp_dir / f"{in_pdf.stem}-div{division}-subsections.zip"
        split_result = split_spec_by_section(
            pdf_path=in_pdf,
            out_zip_path=zip_path,
            division=division,
            section_regex=section_regex
        )
        if not split_result.get("success"):
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({'error': split_result.get("error", "Failed to split spec")}), 500

        JOBS[job_id] = {
            "status": "analyzing",
            "message": "Split completed. Analyzing sections in background.",
            "split_zip": str(zip_path),
            "analysis_zip": None,
            "stats": None,
        }

        sections_dir = _list_sections_dir_from_zip(zip_path)
        analysis_zip_path = temp_dir / f"{in_pdf.stem}-div{division}-analysis.zip"

        def bg(job_id_local: str):
            try:
                stats = _analyze_dir_to_zip(
                    sections_dir,
                    analysis_zip_path,
                    prompt,
                    FIXED_MODEL,
                    MAX_WORKERS
                )
                JOBS[job_id_local]["status"] = "done"
                JOBS[job_id_local]["analysis_zip"] = str(analysis_zip_path)
                JOBS[job_id_local]["message"] = "Analysis complete."
                JOBS[job_id_local]["stats"] = stats
            except Exception as e:
                JOBS[job_id_local]["status"] = "error"
                JOBS[job_id_local]["message"] = f"Analysis error: {e}"

        threading.Thread(target=bg, args=(job_id,), daemon=True).start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "status_url": url_for('job_status', job_id=job_id),
            "split_zip_url": url_for('download_split_zip', job_id=job_id),
            "analysis_zip_url": url_for('download_analysis_zip', job_id=job_id),
        })

    except Exception as e:
        return jsonify({'error': f"Server error: {e}"}), 500

@app.route('/api/job/<job_id>/status')
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'job not found'}), 404
    return jsonify({
        'status': job['status'],
        'message': job['message'],
        'split_ready': bool(job.get('split_zip')),
        'analysis_ready': bool(job.get('analysis_zip')),
        'stats': job.get('stats'),
    })

@app.route('/api/job/<job_id>/split.zip')
def download_split_zip(job_id):
    job = JOBS.get(job_id)
    if not job or not job.get("split_zip"):
        return jsonify({'error': 'job not found or split not ready'}), 404
    return send_file(job["split_zip"], as_attachment=True,
                     download_name=Path(job["split_zip"]).name,
                     mimetype='application/zip')

@app.route('/api/job/<job_id>/analysis.zip')
def download_analysis_zip(job_id):
    job = JOBS.get(job_id)
    if not job or not job.get("analysis_zip"):
        return jsonify({'error': 'job not found or analysis not ready'}), 404
    return send_file(job["analysis_zip"], as_attachment=True,
                     download_name=Path(job["analysis_zip"]).name,
                     mimetype='application/zip')

@app.route('/health')
def health_check():
    return jsonify({'status': 'ok', 'message': 'PDF Tools API is running'})

# ---------------- Entrypoint ----------------
if __name__ == '__main__':
    required = [
        'backend_highlighter.py',
        'backend_spec_splitter.py',
        'backend_spec_analyzer.py',
        'highlighterweb.html'
    ]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print(f"\nCurrent directory: {os.getcwd()}")
        print("Files in current directory:")
        for f in os.listdir('.'):
            print(f"  {f}")
        sys.exit(1)

    print("\nURL Map:\n", app.url_map, "\n")
    print("Starting PDF Tools Web Server…")
    print("Open your browser to: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
