# scripts/preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import pickle  # <-- Added import

def clean_text(text):
    """
    Cleans the input text by removing non-alphabetic characters and extra spaces.
    
    Parameters:
        text (str): Raw text extracted from a resume.
        
    Returns:
        str: Cleaned text.
    """
    # Remove non-alphabetic characters
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """
    Preprocesses the input text by converting to lowercase, tokenizing, removing stopwords, and lemmatizing.
    
    Parameters:
        text (str): Raw text extracted from a resume.
        
    Returns:
        str: Preprocessed text.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Clean and lowercase
    text = clean_text(text.lower())
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(processed_tokens)

def preprocess_texts(extracted_texts):
    """
    Applies preprocessing to all extracted texts.
    
    Parameters:
        extracted_texts (dict): Dictionary mapping file paths to extracted text.
        
    Returns:
        dict: Dictionary mapping file paths to preprocessed text.
    """
    preprocessed_texts = {}
    for file, text in extracted_texts.items():
        preprocessed = preprocess_text(text)
        preprocessed_texts[file] = preprocessed
    return preprocessed_texts

def save_preprocessed_texts(preprocessed_texts, output_path):
    """
    Saves the preprocessed texts to a pickle file.
    
    Parameters:
        preprocessed_texts (dict): Dictionary of preprocessed texts.
        output_path (str): Path to save the pickle file.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(preprocessed_texts, f)

# For testing purposes
if __name__ == "__main__":
    from text_extraction import extract_texts
    from data_collection import get_resume_files

    # Ensure NLTK resources are downloaded
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    input_folder = "input/resumes"
    resumes = get_resume_files(input_folder)
    extracted_texts = extract_texts(resumes)
    preprocessed_texts = preprocess_texts(extracted_texts)
    
    # Save preprocessed texts
    preprocessed_texts_path = "output/preprocessed_texts.pkl"
    save_preprocessed_texts(preprocessed_texts, preprocessed_texts_path)
    
    print(f"Preprocessed {len(preprocessed_texts)} resumes.")
    for file, text in preprocessed_texts.items():
        print(f"\n--- {Path(file).name} ---\n{text[:200]}...")  # Print first 200 characters
