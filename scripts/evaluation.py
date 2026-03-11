#scripts\evaluation.py

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle
import os
from pathlib import Path

def load_cluster_assignments(assignments_path):
    """
    Loads cluster assignments from a pickle file.

    Parameters:
        assignments_path (str): Path to the cluster assignments file.

    Returns:
        dict: Dictionary mapping file paths to their assigned cluster labels.
    """
    with open(assignments_path, 'rb') as f:
        cluster_assignments = pickle.load(f)
    return cluster_assignments

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

def evaluate_clustering(embeddings, cluster_labels):
    """
    Evaluates clustering performance using various metrics.

    Parameters:
        embeddings (list or array): Array of embedding vectors.
        cluster_labels (list or array): List of cluster labels for each embedding.

    Returns:
        dict: Dictionary containing evaluation metric scores.
    """
    silhouette = silhouette_score(embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
    
    metrics = {
        'Silhouette Score': silhouette,
        'Davies-Bouldin Index': davies_bouldin,
        'Calinski-Harabasz Index': calinski_harabasz
    }
    
    return metrics

def display_evaluation_metrics(metrics):
    """
    Prints the evaluation metrics.

    Parameters:
        metrics (dict): Dictionary containing evaluation metric scores.
    """
    print("Clustering Evaluation Metrics:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

def save_evaluation_report(metrics, output_path):
    """
    Saves the evaluation metrics to a text file.

    Parameters:
        metrics (dict): Dictionary containing evaluation metric scores.
        output_path (str): Path to save the evaluation report.
    """
    with open(output_path, 'w') as f:
        f.write("Clustering Evaluation Metrics:\n")
        for metric, score in metrics.items():
            f.write(f"{metric}: {score:.4f}\n")

def evaluation_pipeline(assignments_path, embeddings_path, report_path='../output/evaluation_report.txt'):
    """
    Complete evaluation pipeline.

    Parameters:
        assignments_path (str): Path to the cluster assignments file.
        embeddings_path (str): Path to the embeddings file.
        report_path (str): Path to save the evaluation report.

    Returns:
        dict: Dictionary containing evaluation metric scores.
    """
    cluster_assignments = load_cluster_assignments(assignments_path)
    embeddings_dict = load_embeddings(embeddings_path)
    
    # Ensure that the order of embeddings matches the cluster labels
    files = list(cluster_assignments.keys())
    embeddings = [embeddings_dict[file] for file in files]
    cluster_labels = [cluster_assignments[file] for file in files]
    
    metrics = evaluate_clustering(embeddings, cluster_labels)
    display_evaluation_metrics(metrics)
    save_evaluation_report(metrics, report_path)
    
    return metrics

# For testing purposes
if __name__ == "__main__":
    from clustering import load_embeddings
    
    assignments_path = "output/cluster_assignments.pkl"
    embeddings_path = "output/embeddings.pkl"
    report_path = "output/evaluation_report.txt"
    
    metrics = evaluation_pipeline(assignments_path, embeddings_path, report_path)
