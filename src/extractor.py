"""Extract text from .txt and .pdf files."""
from pathlib import Path

def extract_text(filepath: str) -> str:
    path = Path(filepath)
    if path.suffix.lower() == ".pdf":
        return _extract_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")

def _extract_pdf(path: Path) -> str:
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError("Install pymupdf: pip install pymupdf")
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        pages.append(f"[PAGE {i+1}]\n{page.get_text()}")
    return "\n\n".join(pages)
