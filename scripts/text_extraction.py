
# scripts\text_extraction.py
import os
from pathlib import Path
from docx import Document

def extract_text_from_docx(file_path):
    """
    Extracts text from a .docx Word document.

    Parameters:
        file_path (str): Path to the .docx file.

    Returns:
        str: Extracted text content.
    """
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_texts(resume_files):
    """
    Extracts text from a list of resume files.

    Parameters:
        resume_files (list): List of resume file paths.

    Returns:
        dict: Dictionary mapping file paths to their extracted text.
    """
    texts = {}
    for file in resume_files:
        text = extract_text_from_docx(file)
        if text:
            texts[file] = text
    return texts

# For testing purposes
if __name__ == "__main__":
    from data_collection import get_resume_files

    input_folder = "input/resumes"
    resumes = get_resume_files(input_folder)
    extracted_texts = extract_texts(resumes)
    
    print(f"Extracted text from {len(extracted_texts)} resumes.")
    for file, text in extracted_texts.items():
        print(f"\n--- {Path(file).name} ---\n{text[:200]}...")  # Print first 200 characters
