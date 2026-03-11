# chatbot/store_embeddings.py

import os
import json
from pathlib import Path

import faiss
import numpy as np

def load_embeddings(embeddings_file):
    """
    Loads embeddings from a JSON file.
    
    Parameters:
        embeddings_file (str): Path to the resume_embeddings.json file.
        
    Returns:
        list: List of embedding vectors.
        list: List of metadata dictionaries.
    """
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    embeddings = []
    metadata = []
    
    for resume_name, chunks in data.items():
        for chunk in chunks:
            embeddings.append(chunk['embedding'])
            metadata.append({
                "resume_name": resume_name,
                "chunk_index": chunk['chunk_index'],
                "text": chunk['text'],
                "resume_path": chunk['resume_path']  # Include resume path
            })
    
    return embeddings, metadata

def create_faiss_index(embeddings, dimension, index_path, metadata, metadata_path):
    """
    Creates a FAISS index from embeddings and saves the index and metadata.
    
    Parameters:
        embeddings (list): List of embedding vectors.
        dimension (int): Dimension of the embedding vectors.
        index_path (str): Path to save the FAISS index file.
        metadata (list): List of metadata dictionaries.
        metadata_path (str): Path to save the metadata JSON file.
    """
    # Convert embeddings to numpy array
    embeddings_np = np.array(embeddings).astype('float32')
    
    # Initialize FAISS index
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the index
    index.add(embeddings_np)
    
    # Save the index (convert Path to string)
    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved to '{index_path}'.")
    
    # Save metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to '{metadata_path}'.")

def store_embeddings(input_file, index_output, metadata_output, dimension=384):
    """
    Loads embeddings and stores them in a FAISS index with metadata.
    
    Parameters:
        input_file (str): Path to the resume_embeddings.json file.
        index_output (str): Path to save the FAISS index file.
        metadata_output (str): Path to save the metadata JSON file.
        dimension (int): Dimension of the embedding vectors.
    """
    embeddings, metadata = load_embeddings(input_file)
    create_faiss_index(embeddings, dimension, index_output, metadata, metadata_output)

if __name__ == "__main__":
    input_file = Path("output/resume_embeddings.json")
    index_output = Path("output/resume_index.faiss")
    metadata_output = Path("output/resume_metadata.json")
    
    # Ensure the output directory exists
    index_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    
    store_embeddings(input_file, index_output, metadata_output)
