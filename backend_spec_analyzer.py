# backend_spec_analyzer.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict
import os

# PDF report writer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors

# Try the new SDK first, then fall back to legacy
NEW_SDK = False
try:
    from google import genai as genai_new
    from google.genai import types as genai_types
    NEW_SDK = True
except Exception:
    try:
        import google.generativeai as genai_old
    except Exception as e:
        raise ImportError(
            "Install one of the SDKs:\n"
            "  pip install google-genai  # preferred\n"
            "or\n"
            "  pip install google-generativeai"
        )

DEFAULT_PROMPT = (
    "Extract a structured, human-readable summary of this spec section.\n"
    "- Include: Section number (e.g., 23 20 05), Section name, scope, product list, key specifications, "
    "temperature ratings, voltage/amps/phase, materials/construction, certifications, ratings, "
    "and any submittal requirements.\n"
    "- Be precise. Use bullet points and tables where it clarifies details.\n"
    "- If info is not present, say 'Not specified'."
)

def _write_report_pdf(out_path: Path, title: str, body_text: str) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'], textColor=colors.darkblue, spaceAfter=16
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], leading=14, allowWidows=1, allowOrphans=1
    )
    doc = SimpleDocTemplate(str(out_path), pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 8))
    # Lightly escape angle brackets to avoid ReportLab parsing issues
    safe = body_text.replace("<", "&lt;").replace(">", "&gt;")
    for para in safe.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip().replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 8))
    doc.build(story)

def _generate_with_new_sdk(pdf_bytes: bytes, prompt: str, model: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    client = genai_new.Client(api_key=api_key)
    parts = [
        genai_types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
        prompt  # <-- pass prompt as plain string, do NOT use Part.from_text
    ]
    resp = client.models.generate_content(model=model, contents=parts)
    return (resp.text or "").strip()

def _generate_with_legacy_sdk(pdf_bytes: bytes, prompt: str, model_name: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    genai_old.configure(api_key=api_key)
    model = genai_old.GenerativeModel(model_name)
    # Legacy SDK also accepts a list mixing dict parts and strings
    resp = model.generate_content([
        {"mime_type": "application/pdf", "data": pdf_bytes},
        prompt
    ])
    return (getattr(resp, "text", "") or "").strip()

def analyze_spec_backend(
    pdf_path: Path,
    output_path: Path,
    prompt: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    title: Optional[str] = None,
) -> Dict:
    """
    Analyze one spec-section PDF with Gemini and write a nicely formatted PDF report.

    Returns: {"success": bool, "error": Optional[str]}
    """
    try:
        pdf_path = Path(pdf_path)
        output_path = Path(output_path)
        if not pdf_path.exists():
            return {"success": False, "error": f"Input PDF not found: {pdf_path}"}

        pdf_bytes = pdf_path.read_bytes()
        user_prompt = prompt.strip() if prompt else DEFAULT_PROMPT

        if NEW_SDK:
            text = _generate_with_new_sdk(pdf_bytes, user_prompt, model)
        else:
            text = _generate_with_legacy_sdk(pdf_bytes, user_prompt, model)

        if not text:
            text = "Model returned no text."

        report_title = title or f"Spec Extraction Report — {pdf_path.name}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_report_pdf(output_path, report_title, text)

        return {"success": True}
    except Exception as e:
        # Ensure something is written so upstream ZIP isn't empty
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _write_report_pdf(output_path, f"Analysis Error — {pdf_path.name}", str(e))
        except Exception:
            pass
        return {"success": False, "error": str(e)}
