# # # scripts/generate_summary.py
# # import os
# # import pickle
# # import json  # <-- Added import
# # from pathlib import Path
# # from collections import Counter
# # import matplotlib.pyplot as plt
# # from wordcloud import WordCloud
# # import spacy
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # import markdown2  # Optional: If you plan to convert Markdown to HTML

# # def load_cluster_assignments(assignments_path):
# #     """
# #     Loads cluster assignments from a pickle file.

# #     Parameters:
# #         assignments_path (str): Path to the cluster assignments file.

# #     Returns:
# #         dict: Dictionary mapping file paths to their assigned cluster labels.
# #     """
# #     with open(assignments_path, 'rb') as f:
# #         cluster_assignments = pickle.load(f)
# #     return cluster_assignments

# # def load_preprocessed_texts(preprocessed_texts_path):
# #     """
# #     Loads preprocessed texts from a pickle file.

# #     Parameters:
# #         preprocessed_texts_path (str): Path to the preprocessed texts file.

# #     Returns:
# #         dict: Dictionary mapping file paths to their preprocessed text.
# #     """
# #     with open(preprocessed_texts_path, 'rb') as f:
# #         preprocessed_texts = pickle.load(f)
# #     return preprocessed_texts

# # def extract_top_keywords(texts, top_n=10):
# #     """
# #     Extracts top keywords from a list of texts using TF-IDF.

# #     Parameters:
# #         texts (list): List of text strings.
# #         top_n (int): Number of top keywords to extract.

# #     Returns:
# #         list: List of top keywords.
# #     """
# #     vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
# #     tfidf_matrix = vectorizer.fit_transform(texts)
# #     feature_names = vectorizer.get_feature_names_out()
# #     avg_tfidf = tfidf_matrix.mean(axis=0).A1
# #     tfidf_scores = dict(zip(feature_names, avg_tfidf))
# #     sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
# #     top_keywords = [word for word, score in sorted_keywords[:top_n]]
# #     return top_keywords

# # def extract_common_entities(texts, nlp, top_n=10):
# #     """
# #     Extracts the most common named entities from a list of texts.

# #     Parameters:
# #         texts (list): List of text strings.
# #         nlp: spaCy language model.
# #         top_n (int): Number of top entities to extract.

# #     Returns:
# #         list: List of top entities.
# #     """
# #     entities = []
# #     for doc in nlp.pipe(texts, disable=["parser"]):  # Only NER is needed
# #         entities.extend([ent.text for ent in doc.ents if ent.label_ in [
# #             'ORG', 'PERSON', 'GPE', 'NORP', 'FAC', 'PRODUCT', 
# #             'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']])
# #     counter = Counter(entities)
# #     common_entities = [entity for entity, count in counter.most_common(top_n)]
# #     return common_entities

# # def generate_wordcloud(text, cluster_label, output_folder):
# #     """
# #     Generates and saves a word cloud image for the given text.

# #     Parameters:
# #         text (str): Combined text for the cluster.
# #         cluster_label (str): Label of the cluster.
# #         output_folder (str): Path to save the word cloud image.
# #     """
# #     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
# #     plt.figure(figsize=(10, 5))
# #     plt.imshow(wordcloud, interpolation='bilinear')
# #     plt.axis('off')
# #     plt.title(f'Word Cloud for {cluster_label}', fontsize=16)
# #     plt.tight_layout(pad=0)
# #     image_path = Path(output_folder) / f'{cluster_label}_wordcloud.png'
# #     plt.savefig(image_path)
# #     plt.close()

# # def compile_summary(clusters_info, output_folder, summary_path):
# #     """
# #     Compiles the summary report in Markdown format.

# #     Parameters:
# #         clusters_info (dict): Dictionary containing information about each cluster.
# #         output_folder (str): Path where images are saved.
# #         summary_path (str): Path to save the summary report.
# #     """
# #     summary_md = "# Resume Clustering Summary Report\n\n"
# #     summary_md += f"**Date:** {Path(summary_path).stem}\n\n"
# #     summary_md += "---\n\n"

# #     summary_md += "## Overall Clustering Overview\n\n"
# #     total_clusters = len(clusters_info)
# #     summary_md += f"- **Total Number of Clusters:** {total_clusters}\n\n"
# #     summary_md += "## Detailed Cluster Descriptions\n\n"

# #     for cluster_label, info in clusters_info.items():
# #         summary_md += f"---\n\n"
# #         summary_md += f"### {cluster_label}\n\n"
# #         summary_md += f"- **Number of Resumes:** {info['size']}\n\n"
# #         summary_md += f"- **Top Keywords:** {', '.join(info['top_keywords'])}\n\n"
# #         summary_md += f"- **Common Entities:** {', '.join(info['common_entities'])}\n\n"
# #         summary_md += f"- **Representative Resumes:**\n"
# #         for resume in info['representative_resumes']:
# #             summary_md += f"  - **{resume}**\n"
# #         summary_md += f"\n- **Unique Attributes:**\n  - {info['unique_attributes']}\n\n"
# #         summary_md += f"![Word Cloud for {cluster_label}](./{cluster_label}_wordcloud.png)\n\n"

# #     # Write Markdown to file
# #     with open(summary_path, 'w', encoding='utf-8') as f:
# #         f.write(summary_md)

# #     # Save clusters_info to JSON for later use
# #     clusters_info_path = Path(output_folder) / 'clusters_info.json'
# #     # Convert keys to strings to ensure JSON compatibility
# #     clusters_info_str_keys = {str(k): v for k, v in clusters_info.items()}
# #     with open(clusters_info_path, 'w', encoding='utf-8') as f:
# #         json.dump(clusters_info_str_keys, f, indent=4)

# #     print(f"Cluster summary report generated at '{summary_path}'.")
# #     print(f"Cluster information saved at '{clusters_info_path}'.")

# # def generate_summary(assignments_path, preprocessed_texts_path, output_folder='output', summary_filename='cluster_summary.md'):
# #     """
# #     Generates a summary report for the clustered resumes.

# #     Parameters:
# #         assignments_path (str): Path to the cluster assignments file.
# #         preprocessed_texts_path (str): Path to the preprocessed texts file.
# #         output_folder (str): Path to save the summary report and visualizations.
# #         summary_filename (str): Name of the summary file.
# #     """
# #     cluster_assignments = load_cluster_assignments(assignments_path)
# #     preprocessed_texts = load_preprocessed_texts(preprocessed_texts_path)

# #     # Organize texts by cluster
# #     clusters = {}
# #     for file_path, cluster_label in cluster_assignments.items():
# #         clusters.setdefault(cluster_label, []).append(preprocessed_texts[file_path])

# #     # Load spaCy model
# #     try:
# #         nlp = spacy.load('en_core_web_sm')
# #     except OSError:
# #         print("spaCy model 'en_core_web_sm' not found. Downloading...")
# #         from spacy.cli import download
# #         download('en_core_web_sm')
# #         nlp = spacy.load('en_core_web_sm')

# #     clusters_info = {}

# #     for cluster_label, texts in clusters.items():
# #         combined_text = ' '.join(texts)
# #         top_keywords = extract_top_keywords(texts, top_n=10)
# #         common_entities = extract_common_entities(texts, nlp, top_n=10)
# #         # Select representative resumes (first 2 for simplicity)
# #         representative_resumes = [Path(file).name for file, label in cluster_assignments.items() if label == cluster_label][:2]
# #         unique_attributes = " - ".join(top_keywords[:3])  # Example unique attributes

# #         # Convert cluster_label to string to avoid JSON serialization issues
# #         cluster_label_str = str(cluster_label)

# #         clusters_info[cluster_label_str] = {
# #             'size': len(texts),
# #             'top_keywords': top_keywords,
# #             'common_entities': common_entities,
# #             'representative_resumes': representative_resumes,
# #             'unique_attributes': unique_attributes
# #         }

# #         # Generate word cloud for the cluster
# #         generate_wordcloud(combined_text, cluster_label_str, output_folder)

# #     # Compile the summary report
# #     summary_path = Path(output_folder) / summary_filename
# #     compile_summary(clusters_info, output_folder, summary_path)


# # scripts/generate_summary.py
# import os
# import pickle
# import json  # <-- Added import
# from pathlib import Path
# from collections import Counter
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer
# import markdown2  # Used to convert Markdown to HTML

# def load_cluster_assignments(assignments_path):
#     """
#     Loads cluster assignments from a pickle file.
#     """
#     with open(assignments_path, 'rb') as f:
#         cluster_assignments = pickle.load(f)
#     return cluster_assignments

# def load_preprocessed_texts(preprocessed_texts_path):
#     """
#     Loads preprocessed texts from a pickle file.
#     """
#     with open(preprocessed_texts_path, 'rb') as f:
#         preprocessed_texts = pickle.load(f)
#     return preprocessed_texts

# def extract_top_keywords(texts, top_n=10):
#     """
#     Extracts top keywords from a list of texts using TF-IDF.
#     """
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
#     tfidf_matrix = vectorizer.fit_transform(texts)
#     feature_names = vectorizer.get_feature_names_out()
#     avg_tfidf = tfidf_matrix.mean(axis=0).A1
#     tfidf_scores = dict(zip(feature_names, avg_tfidf))
#     sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
#     top_keywords = [word for word, score in sorted_keywords[:top_n]]
#     return top_keywords

# def extract_common_entities(texts, nlp, top_n=10):
#     """
#     Extracts the most common named entities from a list of texts.
#     """
#     entities = []
#     for doc in nlp.pipe(texts, disable=["parser"]):  # Only NER is needed
#         entities.extend([ent.text for ent in doc.ents if ent.label_ in [
#             'ORG', 'PERSON', 'GPE', 'NORP', 'FAC', 'PRODUCT', 
#             'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']])
#     counter = Counter(entities)
#     common_entities = [entity for entity, count in counter.most_common(top_n)]
#     return common_entities

# def generate_wordcloud(text, cluster_label, output_folder):
#     """
#     Generates and saves a word cloud image for the given text.
#     """
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.title(f'Word Cloud for {cluster_label}', fontsize=16)
#     plt.tight_layout(pad=0)
#     image_path = Path(output_folder) / f'{cluster_label}_wordcloud.png'
#     plt.savefig(image_path)
#     plt.close()

# def compile_summary(clusters_info, output_folder, summary_path):
#     """
#     Compiles the summary report in Markdown format and converts it to HTML.
#     Saves both versions and returns them.
#     """
#     summary_md = "# Resume Clustering Summary Report\n\n"
#     summary_md += f"**Date:** {Path(summary_path).stem}\n\n"
#     summary_md += "---\n\n"
#     summary_md += "## Overall Clustering Overview\n\n"
#     total_clusters = len(clusters_info)
#     summary_md += f"- **Total Number of Clusters:** {total_clusters}\n\n"
#     summary_md += "## Detailed Cluster Descriptions\n\n"

#     for cluster_label, info in clusters_info.items():
#         summary_md += f"---\n\n"
#         summary_md += f"### {cluster_label}\n\n"
#         summary_md += f"- **Number of Resumes:** {info['size']}\n\n"
#         summary_md += f"- **Top Keywords:** {', '.join(info['top_keywords'])}\n\n"
#         summary_md += f"- **Common Entities:** {', '.join(info['common_entities'])}\n\n"
#         summary_md += f"- **Representative Resumes:**\n"
#         for resume in info['representative_resumes']:
#             summary_md += f"  - **{resume}**\n"
#         summary_md += f"\n- **Unique Attributes:**\n  - {info['unique_attributes']}\n\n"
#         summary_md += f"![Word Cloud for {cluster_label}](./{cluster_label}_wordcloud.png)\n\n"

#     # Write the Markdown summary to file
#     with open(summary_path, 'w', encoding='utf-8') as f:
#         f.write(summary_md)

#     # Convert the Markdown summary to HTML
#     summary_html = markdown2.markdown(summary_md)
#     summary_path_html = Path(output_folder) / (Path(summary_path).stem + ".html")
#     with open(summary_path_html, 'w', encoding='utf-8') as f:
#         f.write(summary_html)

#     print(f"Cluster summary report generated at '{summary_path}'.")
#     print(f"HTML version saved at '{summary_path_html}'.")
    
#     return summary_md, summary_html

# def generate_summary(assignments_path, preprocessed_texts_path, output_folder='output', summary_filename='cluster_summary.md'):
#     """
#     Complete feature extraction pipeline for summary generation.
#     Returns both Markdown and HTML summary versions.
#     """
#     cluster_assignments = load_cluster_assignments(assignments_path)
#     preprocessed_texts = load_preprocessed_texts(preprocessed_texts_path)
    
#     # Organize texts by cluster
#     clusters = {}
#     for file_path, cluster_label in cluster_assignments.items():
#         clusters.setdefault(cluster_label, []).append(preprocessed_texts[file_path])
    
#     # Load spaCy model
#     try:
#         nlp = spacy.load('en_core_web_sm')
#     except OSError:
#         print("spaCy model 'en_core_web_sm' not found. Downloading...")
#         from spacy.cli import download
#         download('en_core_web_sm')
#         nlp = spacy.load('en_core_web_sm')
    
#     clusters_info = {}
#     for cluster_label, texts in clusters.items():
#         combined_text = ' '.join(texts)
#         top_keywords = extract_top_keywords(texts, top_n=10)
#         common_entities = extract_common_entities(texts, nlp, top_n=10)
#         representative_resumes = [Path(file).name for file, label in cluster_assignments.items() if label == cluster_label][:2]
#         unique_attributes = " - ".join(top_keywords[:3])
#         cluster_label_str = str(cluster_label)
#         clusters_info[cluster_label_str] = {
#             'size': len(texts),
#             'top_keywords': top_keywords,
#             'common_entities': common_entities,
#             'representative_resumes': representative_resumes,
#             'unique_attributes': unique_attributes
#         }
#         # Generate word cloud for each cluster
#         generate_wordcloud(combined_text, cluster_label_str, output_folder)
    
#     summary_path = Path(output_folder) / summary_filename
#     md, html = compile_summary(clusters_info, output_folder, summary_path)
#     return md, html
import matplotlib
matplotlib.use('Agg')  # <--- ADDED to prevent Tcl_AsyncDelete errors
import matplotlib.pyplot as plt
import os
import pickle
import json
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import markdown2  # Used to convert Markdown to HTML

def load_cluster_assignments(assignments_path):
    with open(assignments_path, 'rb') as f:
        cluster_assignments = pickle.load(f)
    return cluster_assignments

def load_preprocessed_texts(preprocessed_texts_path):
    with open(preprocessed_texts_path, 'rb') as f:
        preprocessed_texts = pickle.load(f)
    return preprocessed_texts

def extract_top_keywords(texts, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    avg_tfidf = tfidf_matrix.mean(axis=0).A1
    tfidf_scores = dict(zip(feature_names, avg_tfidf))
    sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [word for word, score in sorted_keywords[:top_n]]
    return top_keywords

def extract_common_entities(texts, nlp, top_n=10):
    entities = []
    for doc in nlp.pipe(texts, disable=["parser"]):
        entities.extend([ent.text for ent in doc.ents if ent.label_ in [
            'ORG', 'PERSON', 'GPE', 'NORP', 'FAC', 'PRODUCT', 
            'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']])
    counter = Counter(entities)
    common_entities = [entity for entity, count in counter.most_common(top_n)]
    return common_entities

def generate_wordcloud(text, cluster_label, output_folder):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {cluster_label}', fontsize=16)
    plt.tight_layout(pad=0)
    image_path = Path(output_folder) / f'{cluster_label}_wordcloud.png'
    plt.savefig(image_path)
    plt.close()

def compile_summary(clusters_info, output_folder, summary_path):
    summary_md = "# Resume Clustering Summary Report\n\n"
    summary_md += f"**Date:** {Path(summary_path).stem}\n\n"
    summary_md += "---\n\n"
    summary_md += "## Overall Clustering Overview\n\n"
    total_clusters = len(clusters_info)
    summary_md += f"- **Total Number of Clusters:** {total_clusters}\n\n"
    summary_md += "## Detailed Cluster Descriptions\n\n"

    for cluster_label, info in clusters_info.items():
        summary_md += f"---\n\n"
        summary_md += f"### {cluster_label}\n\n"
        summary_md += f"- **Number of Resumes:** {info['size']}\n\n"
        summary_md += f"- **Top Keywords:** {', '.join(info['top_keywords'])}\n\n"
        summary_md += f"- **Common Entities:** {', '.join(info['common_entities'])}\n\n"
        summary_md += f"- **Representative Resumes:**\n"
        for resume in info['representative_resumes']:
            summary_md += f"  - **{resume}**\n"
        summary_md += f"\n- **Unique Attributes:**\n  - {info['unique_attributes']}\n\n"
        summary_md += f"![Word Cloud for {cluster_label}](./{cluster_label}_wordcloud.png)\n\n"

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_md)

    summary_html = markdown2.markdown(summary_md)
    summary_path_html = Path(output_folder) / (Path(summary_path).stem + ".html")
    with open(summary_path_html, 'w', encoding='utf-8') as f:
        f.write(summary_html)

    print(f"Cluster summary report generated at '{summary_path}'.")
    print(f"HTML version saved at '{summary_path_html}'.")
    
    return summary_md, summary_html

def generate_summary(assignments_path, preprocessed_texts_path, output_folder='output', summary_filename='cluster_summary.md'):
    cluster_assignments = load_cluster_assignments(assignments_path)
    preprocessed_texts = load_preprocessed_texts(preprocessed_texts_path)
    
    # Organize texts by cluster
    clusters = {}
    for file_path, cluster_label in cluster_assignments.items():
        clusters.setdefault(cluster_label, []).append(preprocessed_texts[file_path])
    
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("spaCy model 'en_core_web_sm' not found. Downloading...")
        from spacy.cli import download
        download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    
    clusters_info = {}
    for cluster_label, texts in clusters.items():
        combined_text = ' '.join(texts)
        top_keywords = extract_top_keywords(texts, top_n=10)
        common_entities = extract_common_entities(texts, nlp, top_n=10)
        representative_resumes = [Path(file).name for file, label in cluster_assignments.items() if label == cluster_label][:2]
        unique_attributes = " - ".join(top_keywords[:3])
        cluster_label_str = str(cluster_label)
        clusters_info[cluster_label_str] = {
            'size': len(texts),
            'top_keywords': top_keywords,
            'common_entities': common_entities,
            'representative_resumes': representative_resumes,
            'unique_attributes': unique_attributes
        }
        generate_wordcloud(combined_text, cluster_label_str, output_folder)
    
    summary_path = Path(output_folder) / summary_filename
    md, html = compile_summary(clusters_info, output_folder, summary_path)
    return md, html
