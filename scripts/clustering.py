# # scripts\clustering.py

# from sklearn.cluster import KMeans
# import pickle
# import os
# from pathlib import Path

# def load_embeddings(embeddings_path):
#     """
#     Loads embeddings from a pickle file.

#     Parameters:
#         embeddings_path (str): Path to the embeddings file.

#     Returns:
#         dict: Loaded embeddings.
#     """
#     with open(embeddings_path, 'rb') as f:
#         embeddings = pickle.load(f)
#     return embeddings

# def determine_optimal_k(embeddings, max_k=10):
#     """
#     Determines the optimal number of clusters using the Elbow Method.

#     Parameters:
#         embeddings (list or array): Array of embedding vectors.
#         max_k (int): Maximum number of clusters to try.

#     Returns:
#         int: Optimal number of clusters.
#     """
#     from sklearn.metrics import silhouette_score
#     import matplotlib.pyplot as plt

#     wcss = []
#     silhouette_scores = []
#     K = range(2, max_k+1)
#     for k in K:
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         kmeans.fit(embeddings)
#         wcss.append(kmeans.inertia_)
#         score = silhouette_score(embeddings, kmeans.labels_)
#         silhouette_scores.append(score)
    
#     # Plot Elbow Method
#     plt.figure(figsize=(10,5))
#     plt.plot(K, wcss, 'bx-')
#     plt.xlabel('Number of clusters (k)')
#     plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
#     plt.title('Elbow Method For Optimal k')
#     plt.savefig('output/elbow_method.png')
#     plt.close()
    
#     # Plot Silhouette Scores
#     plt.figure(figsize=(10,5))
#     plt.plot(K, silhouette_scores, 'bx-')
#     plt.xlabel('Number of clusters (k)')
#     plt.ylabel('Silhouette Score')
#     plt.title('Silhouette Scores For Various k')
#     plt.savefig('output/silhouette_scores.png')
#     plt.close()
    
#     # Choose k with highest silhouette score
#     optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
#     return optimal_k

# def perform_kmeans_clustering(embeddings, n_clusters):
#     """
#     Performs K-Means clustering on the embeddings.

#     Parameters:
#         embeddings (list or array): Array of embedding vectors.
#         n_clusters (int): Number of clusters.

#     Returns:
#         KMeans: Fitted KMeans model.
#     """
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(embeddings)
#     return kmeans

# def assign_clusters(kmeans, embeddings_dict):
#     """
#     Assigns cluster labels to each resume.

#     Parameters:
#         kmeans (KMeans): Fitted KMeans model.
#         embeddings_dict (dict): Dictionary mapping file paths to embeddings.

#     Returns:
#         dict: Dictionary mapping file paths to their assigned cluster labels.
#     """
#     cluster_assignments = {}
#     for file, embedding in embeddings_dict.items():
#         cluster = kmeans.predict([embedding])[0]
#         cluster_assignments[file] = cluster
#     return cluster_assignments

# def save_cluster_assignments(cluster_assignments, output_path):
#     """
#     Saves the cluster assignments to a pickle file.

#     Parameters:
#         cluster_assignments (dict): Dictionary of cluster assignments.
#         output_path (str): Path to save the cluster assignments file.
#     """
#     with open(output_path, 'wb') as f:
#         pickle.dump(cluster_assignments, f)

# def clustering_pipeline(embeddings_path, save_path='../output/cluster_assignments.pkl', max_k=10):
#     """
#     Complete clustering pipeline.

#     Parameters:
#         embeddings_path (str): Path to the embeddings file.
#         save_path (str): Path to save the cluster assignments.
#         max_k (int): Maximum number of clusters to try for determining optimal k.

#     Returns:
#         dict: Dictionary of cluster assignments.
#         KMeans: Fitted KMeans model.
#     """
#     embeddings_dict = load_embeddings(embeddings_path)
#     files = list(embeddings_dict.keys())
#     embeddings = list(embeddings_dict.values())
    
#     # Determine optimal k
#     optimal_k = determine_optimal_k(embeddings, max_k)
#     print(f"Optimal number of clusters determined: {optimal_k}")
    
#     # Perform K-Means clustering
#     kmeans = perform_kmeans_clustering(embeddings, n_clusters=optimal_k)
    
#     # Assign clusters
#     cluster_assignments = assign_clusters(kmeans, embeddings_dict)
    
#     # Save cluster assignments
#     save_cluster_assignments(cluster_assignments, save_path)
    
#     return cluster_assignments, kmeans

# # For testing purposes
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from feature_extraction import load_embeddings

#     embeddings_path = "output/embeddings.pkl"
#     save_path = "output/cluster_assignments.pkl"

#     cluster_assignments, kmeans = clustering_pipeline(embeddings_path, save_path)

#     print(f"Assigned clusters to {len(cluster_assignments)} resumes.")
#     for file, cluster in list(cluster_assignments.items())[:5]:  # Print first 5 assignments
#         print(f"{Path(file).name}: Cluster {cluster}")
import matplotlib
matplotlib.use('Agg')  # <--- ADDED to prevent Tcl_AsyncDelete errors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import os
from pathlib import Path

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

def determine_optimal_k(embeddings, max_k=10):
    """
    Determines the optimal number of clusters using the Elbow Method.

    Parameters:
        embeddings (list or array): Array of embedding vectors.
        max_k (int): Maximum number of clusters to try.

    Returns:
        int: Optimal number of clusters.
    """
    from sklearn.metrics import silhouette_score

    wcss = []
    silhouette_scores = []
    K = range(2, max_k+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(embeddings, kmeans.labels_)
        silhouette_scores.append(score)
    
    # Plot Elbow Method
    plt.figure(figsize=(10,5))
    plt.plot(K, wcss, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method For Optimal k')
    plt.savefig('output/elbow_method.png')
    plt.close()
    
    # Plot Silhouette Scores
    plt.figure(figsize=(10,5))
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores For Various k')
    plt.savefig('output/silhouette_scores.png')
    plt.close()
    
    # Choose k with highest silhouette score
    optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
    return optimal_k

def perform_kmeans_clustering(embeddings, n_clusters):
    """
    Performs K-Means clustering on the embeddings.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans

def assign_clusters(kmeans, embeddings_dict):
    """
    Assigns cluster labels to each resume.
    """
    cluster_assignments = {}
    for file, embedding in embeddings_dict.items():
        cluster = kmeans.predict([embedding])[0]
        cluster_assignments[file] = cluster
    return cluster_assignments

def save_cluster_assignments(cluster_assignments, output_path):
    """
    Saves the cluster assignments to a pickle file.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(cluster_assignments, f)

def clustering_pipeline(embeddings_path, save_path='../output/cluster_assignments.pkl', max_k=10):
    """
    Complete clustering pipeline.
    """
    embeddings_dict = load_embeddings(embeddings_path)
    files = list(embeddings_dict.keys())
    embeddings = list(embeddings_dict.values())
    
    # Determine optimal k
    optimal_k = determine_optimal_k(embeddings, max_k)
    print(f"Optimal number of clusters determined: {optimal_k}")
    
    # Perform K-Means clustering
    kmeans = perform_kmeans_clustering(embeddings, n_clusters=optimal_k)
    
    # Assign clusters
    cluster_assignments = assign_clusters(kmeans, embeddings_dict)
    
    # Save cluster assignments
    save_cluster_assignments(cluster_assignments, save_path)
    
    return cluster_assignments, kmeans
