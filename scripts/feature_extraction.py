
# scripts\feature_extraction.py
from sentence_transformers import SentenceTransformer
import pickle

def load_model(model_name='all-MiniLM-L6-v2'):
    """
    Loads the pre-trained Sentence-BERT model.

    Parameters:
        model_name (str): Name of the SBERT model to load.

    Returns:
        SentenceTransformer: Loaded SBERT model.
    """
    model = SentenceTransformer(model_name)
    return model

def generate_embeddings(preprocessed_texts, model):
    """
    Generates embeddings for each preprocessed resume text.

    Parameters:
        preprocessed_texts (dict): Dictionary mapping file paths to preprocessed text.
        model (SentenceTransformer): Loaded SBERT model.

    Returns:
        dict: Dictionary mapping file paths to their corresponding embeddings.
    """
    embeddings = {}
    for file, text in preprocessed_texts.items():
        embedding = model.encode(text)
        embeddings[file] = embedding
    return embeddings

def save_embeddings(embeddings, output_path):
    """
    Saves the embeddings to a file using pickle.

    Parameters:
        embeddings (dict): Dictionary of embeddings to save.
        output_path (str): Path to save the embeddings file.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(embeddings_path):
    """
    Loads embeddings from a pickle file.

    Parameters:
        embeddings_path (str): Path to the embeddings file.

    Returns:
        dict: Loaded embeddings.
    """
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def feature_extraction(preprocessed_texts, model_name='all-MiniLM-L6-v2', save_path='../output/embeddings.pkl'):
    """
    Complete feature extraction pipeline.

    Parameters:
        preprocessed_texts (dict): Dictionary mapping file paths to preprocessed text.
        model_name (str): SBERT model name.
        save_path (str): Path to save the embeddings.

    Returns:
        dict: Dictionary of embeddings.
    """
    model = load_model(model_name)
    embeddings = generate_embeddings(preprocessed_texts, model)
    save_embeddings(embeddings, save_path)
    return embeddings

# For testing purposes
if __name__ == "__main__":
    from preprocessing import preprocess_texts
    from text_extraction import extract_texts
    from data_collection import get_resume_files

    # Ensure NLTK resources are downloaded
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    input_folder = "input/resumes"
    resumes = get_resume_files(input_folder)
    extracted_texts = extract_texts(resumes)
    preprocessed_texts = preprocess_texts(extracted_texts)
    
    embeddings = generate_embeddings(preprocessed_texts, load_model())
    print(f"Generated embeddings for {len(embeddings)} resumes.")

    # Optionally, save embeddings
    save_embeddings(embeddings, 'output/embeddings.pkl')
