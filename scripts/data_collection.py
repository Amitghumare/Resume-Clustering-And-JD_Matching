
# scripts\data_collection.py
import os
from pathlib import Path

def get_resume_files(input_folder):
    """
    Collects all Word resume files from the input folder.

    Parameters:
        input_folder (str): Path to the folder containing resume files.

    Returns:
        list: List of file paths to the resume files.
    """
    supported_extensions = ['.doc', '.docx']
    resume_files = []

    input_path = Path(input_folder)
    for file in input_path.iterdir():
        if file.suffix.lower() in supported_extensions:
            resume_files.append(str(file.resolve()))

    return resume_files

# For testing purposes
if __name__ == "__main__":
    input_folder = "input/resumes"
    resumes = get_resume_files(input_folder)
    print(f"Found {len(resumes)} resume files:")
    for resume in resumes:
        print(resume)
