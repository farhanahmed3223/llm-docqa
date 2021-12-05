"""Extract plain text from .txt and .pdf files."""
from pathlib import Path

SUPPORTED = {".txt", ".pdf", ".md"}

def extract_text(filepath: str) -> tuple[str, str]:
    """Return (text, file_type)."""
    path = Path(filepath)
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
        text = page.get_text("text")
        if text.strip():
            out.append(f"[PAGE {i+1}]\n{text.strip()}")
    doc.close()
    return "\n\n".join(out)
