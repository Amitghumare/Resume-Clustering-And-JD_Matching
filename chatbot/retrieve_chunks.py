# chatbot/retrieve_chunks.py

import os
import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path, dimension=384):
    """
    Loads the FAISS index from a file.
    
    Parameters:
        index_path (str): Path to the FAISS index file.
        dimension (int): Dimension of the embedding vectors.
        
    Returns:
        faiss.Index: Loaded FAISS index.
    """
    try:
        index = faiss.read_index(str(index_path))
        return index
    except Exception as e:
        print(f"Error loading FAISS index from '{index_path}': {e}")
        raise

def load_metadata(metadata_path):
    """
    Loads metadata from a JSON file.
    
    Parameters:
        metadata_path (str): Path to the metadata JSON file.
        
    Returns:
        list: List of metadata dictionaries.
    """
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Error loading metadata from '{metadata_path}': {e}")
        raise

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

def generate_query_embedding(query, model):
    """
    Generates an embedding for the user query.
    
    Parameters:
        query (str): The user's query.
        model (SentenceTransformer): The loaded SentenceTransformer model.
        
    Returns:
        np.ndarray: The embedding vector.
    """
    try:
        embedding = model.encode(query)
        return np.array([embedding]).astype('float32')
    except Exception as e:
        print(f"Error generating embedding for query: {e}")
        return None

def search_faiss(index, query_embedding, top_k=5):
    """
    Searches the FAISS index for the top_k most similar embeddings.
    
    Parameters:
        index (faiss.Index): The loaded FAISS index.
        query_embedding (np.ndarray): The embedding vector of the query.
        top_k (int): Number of top similar results to retrieve.
        
    Returns:
        list: List of indices of the top_k similar embeddings.
        list: List of similarity scores for the top_k results.
    """
    try:
        distances, indices = index.search(query_embedding, top_k)
        return indices[0], distances[0]
    except Exception as e:
        print(f"Error searching FAISS index: {e}")
        return [], []

def retrieve_chunks(query, index_path, metadata_path, model_name='all-MiniLM-L6-v2', top_k=5):
    """
    Retrieves the top_k most relevant resume chunks based on the user query.
    
    Parameters:
        query (str): The user's query.
        index_path (str): Path to the FAISS index file.
        metadata_path (str): Path to the metadata JSON file.
        model_name (str): The name of the SentenceTransformer model.
        top_k (int): Number of top similar results to retrieve.
        
    Returns:
        list: List of retrieved text chunks.
    """
    # Load FAISS index
    index = load_faiss_index(index_path)
    
    # Load metadata
    metadata = load_metadata(metadata_path)
    
    # Load model
    model = load_model(model_name)
    
    # Generate query embedding
    query_embedding = generate_query_embedding(query, model)
    if query_embedding is None:
        return []
    
    # Search FAISS index
    indices, distances = search_faiss(index, query_embedding, top_k)
    
    # Retrieve corresponding chunks
    retrieved_chunks = []
    for idx, distance in zip(indices, distances):
        if idx < len(metadata):
            chunk_info = metadata[idx]
            retrieved_chunks.append({
                "resume_name": chunk_info["resume_name"],
                "chunk_index": chunk_info["chunk_index"],
                "text": chunk_info["text"],
                "resume_path": chunk_info["resume_path"],  # Include resume path
                "distance": float(distance)
            })
    
    return retrieved_chunks

# For testing purposes (optional)
if __name__ == "__main__":
    query = "What are the key skills required for a data scientist?"
    index_path = Path("output/resume_index.faiss")
    metadata_path = Path("output/resume_metadata.json")
    top_k = 5
    
    results = retrieve_chunks(query, index_path, metadata_path, top_k=top_k)
    
    print("Retrieved Chunks:")
    for res in results:
        print(f"Resume: {res['resume_name']}, Chunk Index: {res['chunk_index']}, Distance: {res['distance']}")
        print(f"Resume Path: {res['resume_path']}")
        print(f"Text: {res['text']}\n")
