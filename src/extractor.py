"""Extract plain text from .txt, .pdf, and .md files."""
from pathlib import Path

SUPPORTED = {".txt", ".pdf", ".md"}

def extract_text(filepath: str) -> tuple[str, str]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if path.suffix.lower() == ".pdf":
        return _extract_pdf(path), "pdf"
    return path.read_text(encoding="utf-8", errors="ignore"), "text"

def _extract_pdf(path: Path) -> str:
    import fitz
    doc = fitz.open(str(path))
    out = []
    for i, page in enumerate(doc):
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        # guard: skip pages with no text layer (e.g. scanned images)
        if text and text.strip():
            out.append(f"[PAGE {i+1}]\n{text.strip()}")
    doc.close()
    if not out:
        raise ValueError("PDF has no extractable text layer (scanned image?). Try OCR first.")
    return "\n\n".join(out)
