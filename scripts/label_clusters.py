# scripts/label_clusters.py

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

def load_environment_variables():
    """
    Loads environment variables from the .env file.
    """
    load_dotenv()
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set GENAI_API_KEY in your .env file.")
    genai.configure(api_key=api_key)
    return api_key

def load_clusters_info(clusters_info_path):
    """
    Loads cluster information from a JSON file.
    
    Parameters:
        clusters_info_path (str): Path to the clusters_info.json file.
        
    Returns:
        dict: Dictionary containing cluster information.
    """
    with open(clusters_info_path, 'r', encoding='utf-8') as f:
        clusters_info = json.load(f)
    return clusters_info

def sanitize_job_title(job_title):
    """
    Sanitizes the generated job title by removing invalid characters.

    Parameters:
        job_title (str): The job title to sanitize.

    Returns:
        str: A sanitized job title suitable for folder naming.
    """
    # Remove invalid characters: \ / : * ? " < > | &
    sanitized = re.sub(r'[\\/:*?"<>|&]', '', job_title)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Remove any remaining non-printable or control characters
    sanitized = re.sub(r'[\n\r\t]', '', sanitized)
    return sanitized

def is_valid_job_title(job_title):
    """
    Validates the generated job title format.

    Parameters:
        job_title (str): The job title to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    # Simple regex to check for alphanumeric characters and underscores
    pattern = r'^[A-Za-z0-9_]+$'
    return re.match(pattern, job_title) is not None

def generate_job_title(keywords, model):
    """
    Generates a job title based on the provided keywords using a Generative AI model.
    
    Parameters:
        keywords (list): List of top keywords for the cluster.
        model: Configured Generative AI model.
        
    Returns:
        str: Generated and sanitized job title.
    """
    prompt = (
        f"Based on the following keywords: {', '.join(keywords)}, "
        f"generate a single, concise, and descriptive job title suitable for a cluster of resumes. "
        f"The job title should clearly represent the professional role or designation without using special characters (e.g., &, @, #) and without any additional text or formatting."
    )
    try:
        response = model.generate_content(prompt)
        # Take the first line to prevent multi-line responses
        job_title = response.text.strip().split('\n')[0]
        sanitized_job_title = sanitize_job_title(job_title)
        # Validate the job title format
        if not is_valid_job_title(sanitized_job_title):
            print(f"Generated job title '{sanitized_job_title}' is invalid. Using default name.")
            return "Undefined_Job_Title"
        return sanitized_job_title
    except Exception as e:
        print(f"Error generating job title for keywords {keywords}: {e}")
        return "Undefined_Job_Title"

def rename_cluster_folders(clusters_folder, cluster_num, new_label):
    """
    Renames a cluster folder with the new job title label.
    
    Parameters:
        clusters_folder (str): Path to the clusters directory.
        cluster_num (str): Cluster number as a string.
        new_label (str): New descriptive job title.
    """
    old_folder = Path(clusters_folder) / f"Cluster_{cluster_num}"
    new_folder = Path(clusters_folder) / new_label
    try:
        if not old_folder.exists():
            print(f"Cluster folder '{old_folder}' does not exist. Skipping renaming.")
            return
        if new_folder.exists():
            print(f"Destination folder '{new_folder}' already exists. Skipping renaming.")
            return
        old_folder.rename(new_folder)
        print(f"Renamed Cluster_{cluster_num} to {new_label}")
    except Exception as e:
        print(f"Error renaming Cluster_{cluster_num} to {new_label}: {e}")

def save_cluster_labels(cluster_labels, output_folder):
    """
    Saves the cluster labels mapping to a JSON file.
    
    Parameters:
        cluster_labels (dict): Dictionary mapping cluster numbers to job titles.
        output_folder (str): Path to save the labels file.
    """
    labels_path = Path(output_folder) / 'cluster_labels.json'
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_labels, f, indent=4)
    print(f"Cluster labels saved at '{labels_path}'.")

def label_clusters(clusters_info_path, clusters_folder, output_folder='output', model_name="gemini-1.5-flash"):
    """
    Labels each cluster with a descriptive job title using an LLM.
    
    Parameters:
        clusters_info_path (str): Path to the clusters_info.json file.
        clusters_folder (str): Path to the clusters directory containing cluster folders.
        output_folder (str): Path to save the cluster_labels.json file.
        model_name (str): Name of the generative model to use.
    """
    # Load environment variables and configure LLM
    load_environment_variables()
    
    # Initialize the Generative Model after configuring
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error initializing the Generative Model '{model_name}': {e}")
        raise
    
    # Load clusters information
    clusters_info = load_clusters_info(clusters_info_path)
    
    cluster_labels = {}
    
    for cluster_num, info in clusters_info.items():
        keywords = info['top_keywords']
        job_title = generate_job_title(keywords, model)
        cluster_labels[cluster_num] = job_title
        rename_cluster_folders(clusters_folder, cluster_num, job_title)
    
    # Save cluster labels mapping
    save_cluster_labels(cluster_labels, output_folder)

# For testing purposes
if __name__ == "__main__":
    clusters_info_path = "output/clusters_info.json"
    clusters_folder = "output/clusters"
    output_folder = "output"
    model_name = "gemini-1.5-flash"  # Ensure this model is available and you have access
    
    label_clusters(clusters_info_path, clusters_folder, output_folder, model_name)
