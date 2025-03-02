import logging
from pathlib import Path
import PyPDF2  # Make sure to add dependency in requirements


def load_pdf(file_path: str) -> str:
    # TODO: Enhance PDF extraction if needed
    try:
        path = Path(file_path)
        reader = PyPDF2.PdfReader(str(path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        return ""
