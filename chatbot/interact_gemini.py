# chatbot/interact_gemini.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

def load_environment_variables():
    """
    Loads environment variables from the .env file and configures the Gemini API.
    """
    load_dotenv()
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set GENAI_API_KEY in your .env file.")
    genai.configure(api_key=api_key)
    return api_key

def initialize_gemini(model_name="gemini-1.5-flash"):
    """
    Initializes the Gemini Generative Model.
    
    Parameters:
        model_name (str): The name of the Gemini model to use.
        
    Returns:
        GenerativeModel: An instance of the configured Gemini model.
    """
    try:
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        print(f"Error initializing the Generative Model '{model_name}': {e}")
        raise

def generate_response(prompt, model):
    """
    Generates a response from Gemini based on the provided prompt.
    
    Parameters:
        prompt (str): The input text to send to the Gemini model.
        model (GenerativeModel): The initialized Gemini model.
        
    Returns:
        str: The generated response text.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response at this time."

def generate_response_with_chunks(user_query, retrieved_chunks, model=None, model_name="gemini-1.5-flash"):
    """
    Generates a response using Gemini based on the user query and retrieved resume chunks.
    
    Parameters:
        user_query (str): The user's query.
        retrieved_chunks (list): List of retrieved resume chunks.
        model (GenerativeModel, optional): An already initialized Gemini model.
        model_name (str): The name of the Gemini model to use if model is not provided.
        
    Returns:
        str: The generated response from Gemini.
    """
    if model is None:
        model = initialize_gemini(model_name)
    
    # Prepare the context by concatenating retrieved chunks
    context = "\n\n".join([
        f"Resume Path: {chunk['resume_path']}\nResume: {chunk['resume_name']}\nText: {chunk['text']}"
        for chunk in retrieved_chunks
    ])
    
    # Create the prompt
    prompt = (
    f"You are an intelligent assistant that helps users with information from resumes.\n"
    f"If the user's question does not relate to the resumes (e.g., greetings or casual conversation), "
    f"respond politely using your own knowledge and skills.\n\n"
    f"Here are some relevant resume excerpts:\n{context}\n\n"
    f"User's Question: {user_query}\n\n"
    f"Your Answer:"
)

    
    # Generate response
    response = generate_response(prompt, model)
    return response

# Example usage (optional)
if __name__ == "__main__":
    load_environment_variables()
    gemini_model = initialize_gemini()
    
    user_prompt = "Provide a summary of the key skills required for a data scientist."
    response = generate_response(user_prompt, gemini_model)
    print("Gemini Response:", response)
