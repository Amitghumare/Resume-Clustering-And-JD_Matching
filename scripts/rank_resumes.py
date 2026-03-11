# # scripts/rank_resumes.py

# import os
# import pickle
# from pathlib import Path
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import markdown2

# def load_embeddings(embeddings_path):
#     """
#     Loads embeddings from a pickle file.

#     Parameters:
#         embeddings_path (str): Path to the embeddings file.

#     Returns:
#         dict: Dictionary mapping file paths to their embeddings.
#     """
#     with open(embeddings_path, 'rb') as f:
#         embeddings = pickle.load(f)
#     return embeddings

# def load_preprocessed_texts(preprocessed_texts_path):
#     """
#     Loads preprocessed texts from a pickle file.

#     Parameters:
#         preprocessed_texts_path (str): Path to the preprocessed texts file.

#     Returns:
#         dict: Dictionary mapping file paths to their preprocessed text.
#     """
#     with open(preprocessed_texts_path, 'rb') as f:
#         preprocessed_texts = pickle.load(f)
#     return preprocessed_texts

# def preprocess_jd(jd_text, lemmatizer, stop_words):
#     """
#     Preprocesses the job description text.

#     Parameters:
#         jd_text (str): Raw job description text.
#         lemmatizer: NLTK WordNetLemmatizer instance.
#         stop_words (set): Set of stopwords.

#     Returns:
#         str: Preprocessed JD text.
#     """
#     import re
#     import nltk
#     # Remove non-alphabetic characters and lowercase
#     jd_text = re.sub(r'[^A-Za-z\s]', '', jd_text).lower()
#     # Tokenize
#     tokens = nltk.word_tokenize(jd_text)
#     # Remove stopwords and lemmatize
#     processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
#     return ' '.join(processed_tokens)

# def rank_resumes(jd_path, embeddings_path, preprocessed_texts_path, output_folder='output', summary_filename='jd_rankings.md'):
#     """
#     Ranks resumes based on their relevance to the provided job description.

#     Parameters:
#         jd_path (str): Path to the JD text file.
#         embeddings_path (str): Path to the embeddings file.
#         preprocessed_texts_path (str): Path to the preprocessed texts file.
#         output_folder (str): Path to save the rankings.
#         summary_filename (str): Name of the summary file.
#     """
#     import nltk
#     from nltk.corpus import stopwords
#     from nltk.stem import WordNetLemmatizer

#     # Ensure NLTK resources are downloaded
#     nltk.download('punkt')
#     nltk.download('stopwords')
#     nltk.download('wordnet')

#     # Load data
#     embeddings = load_embeddings(embeddings_path)
#     preprocessed_texts = load_preprocessed_texts(preprocessed_texts_path)

#     # Read JD
#     with open(jd_path, 'r', encoding='utf-8') as f:
#         jd_text = f.read()

#     # Preprocess JD
#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words('english'))
#     preprocessed_jd = preprocess_jd(jd_text, lemmatizer, stop_words)

#     # Load SentenceTransformer model
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     # Generate embedding for JD
#     jd_embedding = model.encode(preprocessed_jd).reshape(1, -1)

#     # Prepare resume embeddings
#     resume_paths = list(embeddings.keys())
#     resume_embeddings = [embeddings[resume] for resume in resume_paths]

#     # Compute cosine similarities
#     similarities = cosine_similarity(jd_embedding, resume_embeddings)[0]

#     # Pair resumes with their similarity scores
#     resume_scores = list(zip(resume_paths, similarities))

#     # Sort resumes based on similarity scores in descending order
#     ranked_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)

#     # Generate summary
#     summary_md = f"# Resume Rankings for Job Description: {Path(jd_path).stem}\n\n"
#     summary_md += "---\n\n"
#     summary_md += f"**Job Description File:** {Path(jd_path).name}\n\n"
#     summary_md += f"## Ranked Resumes\n\n"

#     summary_md += "| Rank | Resume | Similarity Score |\n"
#     summary_md += "|------|--------|------------------|\n"

#     top_n = 10  # Number of top resumes to display
#     for rank, (resume, score) in enumerate(ranked_resumes[:top_n], start=1):
#         summary_md += f"| {rank} | {Path(resume).name} | {score:.4f} |\n"

#     # Optionally, include more details or links to resumes
#     summary_md += "\n---\n\n"
#     summary_md += f"**Top {top_n} Resumes are listed above based on their relevance to the provided job description.**\n"

#     # Save summary to Markdown file
#     summary_path = Path(output_folder) / summary_filename
#     with open(summary_path, 'w', encoding='utf-8') as f:
#         f.write(summary_md)

#     print(f"Resume ranking report generated at '{summary_path}'.")

# # For testing purposes
# if __name__ == "__main__":
#     jd_path = "input/jd.txt"  # Path to your JD text file
#     embeddings_path = "output/embeddings.pkl"
#     preprocessed_texts_path = "output/preprocessed_texts.pkl"
#     output_folder = "output"
#     summary_filename = "jd_rankings.md"

#     rank_resumes(jd_path, embeddings_path, preprocessed_texts_path, output_folder, summary_filename)


# scripts/rank_resumes.py

import os
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def load_embeddings(embeddings_path):
    """
    Loads embeddings from a pickle file.

    Parameters:
        embeddings_path (str): Path to the embeddings file.

    Returns:
        dict: Dictionary mapping file paths to their embeddings.
    """
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def load_preprocessed_texts(preprocessed_texts_path):
    """
    Loads preprocessed texts from a pickle file.

    Parameters:
        preprocessed_texts_path (str): Path to the preprocessed texts file.

    Returns:
        dict: Dictionary mapping file paths to their preprocessed text.
    """
    with open(preprocessed_texts_path, 'rb') as f:
        preprocessed_texts = pickle.load(f)
    return preprocessed_texts

def preprocess_jd(jd_text, lemmatizer, stop_words):
    """
    Preprocesses the job description text.

    Parameters:
        jd_text (str): Raw job description text.
        lemmatizer: NLTK WordNetLemmatizer instance.
        stop_words (set): Set of stopwords.

    Returns:
        str: Preprocessed JD text.
    """
    import re
    import nltk
    jd_text = re.sub(r'[^A-Za-z\s]', '', jd_text).lower()
    tokens = nltk.word_tokenize(jd_text)
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(processed_tokens)

def rank_resumes(
    jd_path,
    embeddings_path,
    preprocessed_texts_path,
    output_folder='output',
    summary_filename='jd_rankings.md',
    return_data=False
):
    """
    Ranks resumes based on their relevance to the provided job description and
    optionally returns structured data for web UI rendering.

    Parameters:
        jd_path (str): Path to the JD text file.
        embeddings_path (str): Path to the embeddings file (pickle).
        preprocessed_texts_path (str): Path to the preprocessed texts file (pickle).
        output_folder (str): Path to save the rankings (Markdown).
        summary_filename (str): Name of the summary file (Markdown).
        return_data (bool): If True, returns a structured list of ranked resumes.
                           Otherwise, only writes to a Markdown file and prints.

    Returns:
        list or None: If return_data is True, returns a list of dicts with ranking info.
                      If False, writes a Markdown file to disk and returns None.
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Load data
    embeddings = load_embeddings(embeddings_path)
    preprocessed_texts = load_preprocessed_texts(preprocessed_texts_path)

    # Read JD text
    with open(jd_path, 'r', encoding='utf-8') as f:
        jd_text = f.read()

    # Preprocess JD
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    preprocessed_jd = preprocess_jd(jd_text, lemmatizer, stop_words)

    # Load model and compute JD embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_embedding = model.encode(preprocessed_jd).reshape(1, -1)

    # Prepare resume embeddings
    resume_paths = list(embeddings.keys())
    resume_embeddings = [embeddings[resume] for resume in resume_paths]

    # Compute cosine similarities
    similarities = cosine_similarity(jd_embedding, resume_embeddings)[0]

    # Pair resumes with their similarity scores and sort
    resume_scores = list(zip(resume_paths, similarities))
    ranked_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)

    # Build a structured list (for easy UI rendering)
    structured_rankings = []
    for rank, (resume, score) in enumerate(ranked_resumes, start=1):
        structured_rankings.append({
            "rank": rank,
            "resume_file": Path(resume).name,
            "similarity_score": round(score, 4)
        })

    # Generate Markdown summary
    jd_name = Path(jd_path).stem
    summary_md = f"# Resume Rankings for Job Description: {jd_name}\n\n"
    summary_md += "---\n\n"
    summary_md += f"**Job Description File:** {Path(jd_path).name}\n\n"
    summary_md += "## Ranked Resumes\n\n"
    summary_md += "| Rank | Resume | Similarity Score |\n"
    summary_md += "|------|--------|------------------|\n"
    for item in structured_rankings[:10]:
        summary_md += f"| {item['rank']} | {item['resume_file']} | {item['similarity_score']} |\n"
    summary_md += "\n---\n\n"
    summary_md += "**Top 10 Resumes are listed above based on their relevance.**\n"

    # Save summary to Markdown file
    os.makedirs(output_folder, exist_ok=True)
    summary_path = Path(output_folder) / summary_filename
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_md)

    print(f"Resume ranking report generated at '{summary_path}'.")

    # Return structured data if needed
    if return_data:
        return structured_rankings
    return None

# For direct testing:
if __name__ == "__main__":
    jd_path = "input/jd.txt"
    embeddings_path = "output/embeddings.pkl"
    preprocessed_texts_path = "output/preprocessed_texts.pkl"
    output_folder = "output"
    summary_filename = "jd_rankings.md"

    # Example usage with return_data=True for web usage
    results = rank_resumes(
        jd_path,
        embeddings_path,
        preprocessed_texts_path,
        output_folder,
        summary_filename,
        return_data=True
    )
    print("Top results:", results[:5] if results else "No results")
