# chatbot\generate_embeddings.py

import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

def load_model(model_name='all-MiniLM-L6-v2'):
    """
    Loads the SentenceTransformer model.
    
    Parameters:
        model_name (str): The name of the model to load.
        
    Returns:
        SentenceTransformer: The loaded model.
    """
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        raise

def generate_embedding(text, model):
    """
    Generates an embedding for the given text using SentenceTransformer.
    
    Parameters:
        text (str): The text to generate an embedding for.
        model (SentenceTransformer): The loaded SentenceTransformer model.
        
    Returns:
        list: The embedding vector.
    """
    try:
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def generate_embeddings(input_file, output_file, model):
    """
    Generates embeddings for all resume chunks and saves them to a JSON file.
    
    Parameters:
        input_file (str): Path to the resume_chunks.json file.
        output_file (str): Path to save the resume_embeddings.json file.
        model (SentenceTransformer): The loaded SentenceTransformer model.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        resume_chunks = json.load(f)
    
    resume_embeddings = {}
    
    for resume_name, chunks in resume_chunks.items():
        resume_embeddings[resume_name] = []
        for chunk in chunks:
            embedding = generate_embedding(chunk['text'], model)
            if embedding:
                resume_embeddings[resume_name].append({
                    "chunk_index": chunk['chunk_index'],
                    "text": chunk['text'],
                    "embedding": embedding,
                    "resume_path": chunk['resume_path']  # Retain the resume path
                })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(resume_embeddings, f, indent=4)
    
    print(f"Embeddings generated and saved to '{output_file}'.")

if __name__ == "__main__":
    model = load_model()
    
    input_file = Path("output/resume_chunks.json")
    output_file = Path("output/resume_embeddings.json")
    
    generate_embeddings(input_file, output_file, model)
