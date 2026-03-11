# scripts/interact_gemini.py

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

# Example usage
if __name__ == "__main__":
    load_environment_variables()
    gemini_model = initialize_gemini()
    
    user_prompt = "Provide a summary of the key skills required for a data scientist."
    response = generate_response(user_prompt, gemini_model)
    print("Gemini Response:", response)
