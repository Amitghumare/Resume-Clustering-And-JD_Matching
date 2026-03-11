# chatbot/parse_chunk_resumes.py

import os
from pathlib import Path
from docx import Document
import json

def extract_text_from_docx(file_path):
    """
    Extracts text from a .docx file.
    
    Parameters:
        file_path (str): Path to the .docx file.
        
    Returns:
        str: Extracted text.
    """
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def chunk_text(text, max_length=500):
    """
    Splits text into chunks of a specified maximum length.
    
    Parameters:
        text (str): The text to split.
        max_length (int): Maximum number of characters per chunk.
        
    Returns:
        list: List of text chunks.
    """
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if not para.strip():
            continue  # Skip empty paragraphs
        if len(current_chunk) + len(para) + 1 <= max_length:
            current_chunk += para + '\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + '\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def parse_and_chunk_resumes(input_folder, output_file, max_length=500):
    """
    Parses all .docx resumes in the input folder and splits them into chunks.
    
    Parameters:
        input_folder (str): Path to the folder containing resume .docx files.
        output_file (str): Path to save the output JSON with chunks.
        max_length (int): Maximum number of characters per chunk.
    """
    resumes_path = Path(input_folder)
    all_chunks = {}
    
    for resume_file in resumes_path.glob("*.docx"):
        resume_path = resume_file.resolve()
        resume_name = resume_file.stem
        text = extract_text_from_docx(resume_file)
        chunks = chunk_text(text, max_length)
        all_chunks[resume_name] = []
        for idx, chunk in enumerate(chunks):
            all_chunks[resume_name].append({
                "chunk_index": idx,
                "text": chunk,
                "resume_path": str(resume_path)  # Adding the resume path
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4)
    
    print(f"Parsed and chunked resumes saved to '{output_file}'.")

if __name__ == "__main__":
    input_folder = Path("input/resumes")
    output_file = Path("output/resume_chunks.json")
    max_length = 500  # Adjust this value for smaller or larger chunks
    
    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    parse_and_chunk_resumes(input_folder, output_file, max_length)
