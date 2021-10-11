import os
from pathlib import Path


def extract_text(file_path: str) -> tuple[str, str]:
    """
    Extract plain text from a PDF or .txt file.

    Returns:
        (text, file_type) where file_type is 'pdf' or 'txt'.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file type is unsupported or the PDF appears to be scanned.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(file_path)

    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            raise ValueError("Text file is empty.")
        return text, "txt"

    elif suffix == ".pdf":
        return _extract_pdf(file_path)

    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported types: .pdf, .txt"
        )


def _extract_pdf(file_path: str) -> tuple[str, str]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is required for PDF support. Run: pip install pymupdf")

    doc = fitz.open(file_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if page_text.strip():
            # Tag each page so chunk metadata can reference page numbers later
            pages.append(f"[PAGE {page_num}]\n{page_text.strip()}")

    doc.close()

    if not pages:
        raise ValueError(
            "No extractable text found in PDF. "
            "Scanned/image-only PDFs are not supported — "
            "the file must contain a real text layer."
        )

    full_text = "\n\n".join(pages)
    return full_text, "pdf"
