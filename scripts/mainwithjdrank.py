# scripts/main.py

import os
from pathlib import Path

# Import functions from each module
from data_collection import get_resume_files
from text_extraction import extract_texts
from preprocessing import preprocess_texts, save_preprocessed_texts
from feature_extraction import feature_extraction
from clustering import clustering_pipeline
from evaluation import evaluation_pipeline
from organize_clusters import organize_clusters
from generate_summary import generate_summary
from rank_resumes import rank_resumes  # <-- Added import

def main(input_folder='input/resumes', output_folder='output', jd_path='input/jd.txt'):
    """
    Executes the complete resume clustering and ranking pipeline.

    Parameters:
        input_folder (str): Path to the folder containing resume files.
        output_folder (str): Path to the folder to save outputs.
        jd_path (str): Path to the job description text file.
    """
    # Ensure output directory exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print("=== Data Collection ===")
    resumes = get_resume_files(input_folder)
    print(f"Found {len(resumes)} resumes.")
    
    print("\n=== Text Extraction ===")
    extracted_texts = extract_texts(resumes)
    print(f"Extracted text from {len(extracted_texts)} resumes.")
    
    print("\n=== Preprocessing ===")
    preprocessed_texts = preprocess_texts(extracted_texts)
    print(f"Preprocessed {len(preprocessed_texts)} resumes.")
    
    # Save preprocessed texts
    preprocessed_texts_path = os.path.join(output_folder, 'preprocessed_texts.pkl')
    save_preprocessed_texts(preprocessed_texts, preprocessed_texts_path)
    print(f"Saved preprocessed texts to '{preprocessed_texts_path}'.")
    
    print("\n=== Feature Extraction ===")
    embeddings = feature_extraction(preprocessed_texts, save_path=os.path.join(output_folder, 'embeddings.pkl'))
    print(f"Generated embeddings for {len(embeddings)} resumes.")
    
    print("\n=== Clustering ===")
    cluster_assignments, kmeans_model = clustering_pipeline(
        embeddings_path=os.path.join(output_folder, 'embeddings.pkl'),
        save_path=os.path.join(output_folder, 'cluster_assignments.pkl'),
        max_k=10
    )
    print(f"Assigned clusters to {len(cluster_assignments)} resumes.")
    
    print("\n=== Evaluation ===")
    metrics = evaluation_pipeline(
        assignments_path=os.path.join(output_folder, 'cluster_assignments.pkl'),
        embeddings_path=os.path.join(output_folder, 'embeddings.pkl'),
        report_path=os.path.join(output_folder, 'evaluation_report.txt')
    )
    
    print("\n=== Organizing Clusters ===")
    organize_clusters(
        assignments_path=os.path.join(output_folder, 'cluster_assignments.pkl'),
        embeddings_path=os.path.join(output_folder, 'embeddings.pkl'),
        output_folder=os.path.join(output_folder, 'clusters')  # Specify the clusters directory
    )
    
    print("\n=== Generating Cluster Summary ===")
    generate_summary(
        assignments_path=os.path.join(output_folder, 'cluster_assignments.pkl'),
        preprocessed_texts_path=preprocessed_texts_path,
        output_folder=output_folder,
        summary_filename='cluster_summary.md'
    )
    
    print("\n=== Ranking Resumes Based on Job Description ===")
    rank_resumes(
        jd_path=jd_path,
        embeddings_path=os.path.join(output_folder, 'embeddings.pkl'),
        preprocessed_texts_path=preprocessed_texts_path,
        output_folder=output_folder,
        summary_filename='jd_rankings.md'
    )
    
    print("\n=== Pipeline Completed ===")
    print("Evaluation Metrics:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()
