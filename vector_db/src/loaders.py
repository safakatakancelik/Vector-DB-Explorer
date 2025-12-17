import os
from typing import Optional
from pypdf import PdfReader

class FileLoader:
    @staticmethod
    def load_file(file_path: str) -> Optional[str]:
        """
        Loads text from a file based on extension.
        Supports .txt and .pdf
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.txt':
            return FileLoader._load_txt(file_path)
        elif ext == '.pdf':
            return FileLoader._load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @staticmethod
    def _load_txt(path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def _load_pdf(path: str) -> str:
        reader = PdfReader(path)
        text = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text.append(content)
        return "\n".join(text)
