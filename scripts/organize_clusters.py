# scripts\organize_clusters.py

import os
import shutil
import pickle
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

def create_cluster_folders(output_folder, num_clusters):
    """
    Creates folders for each cluster within the output directory.

    Parameters:
        output_folder (str): Path to the output directory.
        num_clusters (int): Number of clusters.

    Returns:
        dict: Dictionary mapping cluster labels to their folder paths.
    """
    cluster_folders = {}
    for cluster in range(num_clusters):
        cluster_folder = Path(output_folder) / f"Cluster_{cluster}"
        cluster_folder.mkdir(parents=True, exist_ok=True)
        cluster_folders[cluster] = cluster_folder
    return cluster_folders

def move_resumes(cluster_assignments, cluster_folders):
    """
    Moves each resume file into its corresponding cluster folder.

    Parameters:
        cluster_assignments (dict): Dictionary mapping file paths to cluster labels.
        cluster_folders (dict): Dictionary mapping cluster labels to folder paths.
    """
    for file_path, cluster_label in cluster_assignments.items():
        try:
            src = Path(file_path)
            dst = cluster_folders[cluster_label] / src.name
            shutil.copy(src, dst)  # Use shutil.move(src, dst) to move instead of copy
        except Exception as e:
            print(f"Error moving {file_path} to Cluster_{cluster_label}: {e}")

def organize_clusters(assignments_path, embeddings_path, output_folder='../output'):
    """
    Complete pipeline to organize resumes into cluster folders.

    Parameters:
        assignments_path (str): Path to the cluster assignments file.
        embeddings_path (str): Path to the embeddings file.
        output_folder (str): Path to the folder where clusters will be organized.
    """
    cluster_assignments = load_cluster_assignments(assignments_path)
    
    # Determine the number of clusters
    clusters = set(cluster_assignments.values())
    num_clusters = len(clusters)
    print(f"Number of clusters to organize: {num_clusters}")
    
    # Create cluster folders
    cluster_folders = create_cluster_folders(output_folder, num_clusters)
    
    # Move resumes to respective cluster folders
    move_resumes(cluster_assignments, cluster_folders)
    print(f"Resumes have been organized into {num_clusters} cluster folders in '{output_folder}'.")

# For testing purposes
if __name__ == "__main__":
    from clustering import determine_optimal_k

    assignments_path = "output/cluster_assignments.pkl"
    embeddings_path = "output/embeddings.pkl"
    output_folder = "output/clusters"

    organize_clusters(assignments_path, embeddings_path, output_folder)
