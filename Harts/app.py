from flask import Flask, request, jsonify, send_file, render_template_string
import tempfile
import os
from pathlib import Path
import sys

# Make local backends importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Backends ---
try:
    from backend_highlighter import highlight_pdf_backend
except ImportError:
    print("ERROR: Could not import backend_highlighter.py")
    print("Make sure backend_highlighter.py is in the same directory as app.py")
    sys.exit(1)

try:
    from backend_spec_splitter import split_spec_by_section
except ImportError:
    print("ERROR: Could not import backend_spec_splitter.py")
    print("Make sure backend_spec_splitter.py is in the same directory as app.py")
    sys.exit(1)

app = Flask(__name__)
# Allow large uploads (adjust size as needed)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

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

# ---------------- Routes ----------------
@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template_string(get_html_content())

@app.route('/api/get-default-keywords')
def get_default_keywords():
    """Return default keywords from plan_keywords.txt if available."""
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
    """Highlight keywords in an uploaded PDF and return the processed file."""
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file uploaded'}), 400
        pdf_file = request.files['pdf']
        keywords = (request.form.get('keywords') or '').strip()

        if not pdf_file.filename:
            return jsonify({'error': 'No file selected'}), 400
        if not keywords:
            return jsonify({'error': 'No keywords provided'}), 400

        # Temp workspace
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
    """
    Split a large spec PDF by Division and sub-section.
    Expects:
      - file field: 'pdf'
      - form fields: 'division' (e.g., '23'), optional 'section_regex'
    Returns: a ZIP containing the split PDFs.
    """
    try:
        pdf_file = request.files.get('pdf')
        if not pdf_file or not pdf_file.filename:
            return jsonify({'error': 'No spec PDF provided'}), 400

        division = (request.form.get('division', '23') or '23').strip()
        if not division.isdigit() or len(division) != 2:
            return jsonify({'error': "Division must be 2 digits (e.g., '23')"}), 400

        section_regex = (request.form.get('section_regex', '') or '').strip()
        if not section_regex:
            section_regex = r'^\s*SECTION\s+(\d{2})\s+(\d{2})\s+(\d{2})(?:\s*[-–—]?\s*(.+))?\s*$'


        # Save uploaded spec to a temp folder
        temp_dir = tempfile.mkdtemp()
        input_path = Path(temp_dir) / pdf_file.filename
        pdf_file.save(str(input_path))

        # Output zip path
        zip_path = input_path.with_name(f"{input_path.stem}-div{division}-subsections.zip")

        # Run splitter
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

@app.route('/health')
def health_check():
    return jsonify({'status': 'ok', 'message': 'PDF Tools API is running'})

# ---------------- Entrypoint ----------------
if __name__ == '__main__':
    required = ['backend_highlighter.py', 'backend_spec_splitter.py', 'highlighterweb.html']
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

    print("Starting PDF Tools Web Server…")
    print("Open your browser to: http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
