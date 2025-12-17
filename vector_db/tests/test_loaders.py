import pytest
import os
from src.loaders import FileLoader

def test_txt_loader(tmp_path):
    # Create temp txt file
    f = tmp_path / "test.txt"
    content = "Hello world from text file."
    f.write_text(content, encoding='utf-8')
    
    loaded = FileLoader.load_file(str(f))
    assert loaded == content

def test_missing_file():
    with pytest.raises(FileNotFoundError):
        FileLoader.load_file("nonexistent.txt")

def test_unsupported_extension(tmp_path):
    f = tmp_path / "test.xyz"
    f.write_text("content", encoding='utf-8')
    
    with pytest.raises(ValueError, match="Unsupported file extension"):
        FileLoader.load_file(str(f))

# PDF testing would require a valid PDF binary, skipping for simple unit test unless we mock pypdf
