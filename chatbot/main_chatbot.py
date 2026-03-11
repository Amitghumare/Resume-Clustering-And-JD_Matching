# chatbot/main_chatbot.py

import os
from pathlib import Path
from retrieve_chunks import retrieve_chunks
from interact_gemini import load_environment_variables, generate_response_with_chunks

def main():
    """
    Main function to handle user queries and generate responses based on resume data.
    """
    # Load environment variables and configure Gemini API
    try:
        load_environment_variables()
    except ValueError as ve:
        print(ve)
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return
    
    # Paths to necessary files
    project_root = Path(__file__).parent.parent.resolve()
    index_path = project_root / "output" / "resume_index.faiss"
    metadata_path = project_root / "output" / "resume_metadata.json"
    
    # Check if index and metadata files exist
    if not index_path.exists():
        print(f"FAISS index file '{index_path}' does not exist. Please run 'store_embeddings.py' first.")
        return
    if not metadata_path.exists():
        print(f"Metadata file '{metadata_path}' does not exist. Please run 'store_embeddings.py' first.")
        return
    
    # Initialize Gemini model
    try:
        gemini_model = initialize_gemini()
    except Exception as e:
        print(f"Failed to initialize Gemini model: {e}")
        return
    
    print("Welcome to the Resume Chatbot! Ask any question related to the resumes.")
    print("Type 'exit' to quit.\n")
    
    while True:
        # Get user input
        user_query = input("You: ").strip()
        if user_query.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        if not user_query:
            print("Chatbot: Please enter a valid query.")
            continue
        
        # Retrieve relevant chunks
        retrieved_chunks = retrieve_chunks(
            query=user_query,
            index_path=index_path,
            metadata_path=metadata_path,
            model_name='all-MiniLM-L6-v2',  # Ensure consistency with your embedding model
            top_k=5  # Adjust as needed
        )
        
        if not retrieved_chunks:
            print("Chatbot: No relevant information found in the resumes.")
            continue
        
        # Generate response using Gemini
        response = generate_response_with_chunks(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
            model=gemini_model
        )
        
        # Display the response
        print(f"Chatbot: {response}\n")

if __name__ == "__main__":
    # Import the initialize_gemini function here to avoid circular imports
    from interact_gemini import initialize_gemini
    main()
