import sklearn.metrics
from sklearn.metrics import silhouette_score
import sys
import numpy as np
import mysymnmfsp as s  # C module

# initializing the random function
np.random.seed(0)

def read_data(input_data):
    """
    Reads data from a text file and converts it into a list of data points.

    Parameters:
    input_data (str): Path to the input data file.

    Returns:
    list of list of float: A list where each element is a list representing a data point.
    """
    datapoints = []
    with open(input_data, 'r') as file:
        for line in file:
            row = [float(point) for point in line.split(',')]
            datapoints.append(row)
    
    return datapoints

def k_means(victors, K):
    """
    Performs K-means clustering on the given data points.

    Parameters:
    victors (list of list of float): List of data points.
    K (int): Number of clusters.

    Returns:
    list of int: Cluster labels for each data point.
    """
    iter = 300
    epsilon = 0.001
    d = len(victors[0])
    centroids = [victors[i] for i in range(K)]
    clusters = [[] for k in range(K)]
    for iteration in range(iter):
        ## classify the victors to the clusters
        for vic in victors:
            clusters[find_closest_centroid_index(vic,centroids, K,d)].append(vic)
        ## update the centroids ##
        accuracy = True
        for i in range(K):
            old_centroid = centroids[i]
            centroids[i] = mean(d,clusters[i])
            if distance(old_centroid,centroids[i],d) > epsilon:
                accuracy = False
        clusters = [[] for k in range(K)]
        if accuracy:
            break

    kmeans_lab = []
    for vic in victors:
        kmeans_lab.append(find_closest_centroid_index(vic,centroids, K,d))

    return kmeans_lab

def find_closest_centroid_index(victor, centroids, K,d):
    """
    Finds the index of the closest centroid to a given data point.

    Parameters:
    victor (list of float): A data point.
    centroids (list of list of float): List of centroids.
    K (int): Number of clusters.
    d (int): Dimensionality of the data points.

    Returns:
    int: Index of the closest centroid.
    """
    index = -1 #the index of the closest centroid
    min_d = distance(victor,centroids[0],d)
    for i in range(K):
        if (distance(victor,centroids[i],d) <= min_d):
            min_d = distance(victor,centroids[i],d)
            index = i
    return index

def mean(d,cluster): 
    """
    Calculates the centroid of a cluster.

    Parameters:
    d (int): Dimensionality of the data points.
    cluster (list of list of float): List of data points in the cluster.

    Returns:
    list of float: Centroid of the cluster.
    """
    result = [0.0]*d
    for i in range(d):
        for x in cluster:
            result[i] += float(x[i])
        result[i] *= 1/(len(cluster))
    return result

def distance(p,q,d):
    """
    Calculates the Euclidean distance between two data points.

    Parameters:
    p (list of float): First data point.
    q (list of float): Second data point.
    d (int): Dimensionality of the data points.

    Returns:
    float: Euclidean distance between p and q.
    """
    sum = 0
    for i in range(d):
        sum += pow((float(p[i]) - float(q[i])),2)
    return (pow(sum, 0.5))

def symnmf_clusters(datapoints, N, K):
    """
    Performs clustering using Symmetric Non-negative Matrix Factorization (SymNMF).

    Parameters:
    datapoints (list of list of float): List of data points.
    N (int): Number of data points.
    K (int): Number of clusters.

    Returns:
    list of int: Cluster labels for each data point.
    """
    W = np.array(s.norm(datapoints))
    m = np.mean(W)
    H = [[0 for i in range(K)] for j in range(N)]
    for i in range(N):
        for j in range(K):
            H[i][j] = np.random.uniform(0, 2*np.sqrt(m/K))
    symnmf_matrix = s.symnmf(H, W.tolist(), N, K)
    symnmf_matrix = np.array(symnmf_matrix)
    symnmf_labels = np.argmax(symnmf_matrix, axis=1)
    return symnmf_labels


def main():
    """
    Main function to execute the clustering algorithms and print the silhouette scores.

    Takes command line arguments for the number of clusters (K) and the input data file.
    """
    ## assertions ##
    try:
        if len(sys.argv) == 3:
            K = int(sys.argv[1])
            input_data = sys.argv[2]
            if not (input_data.endswith(".txt")):
                print("An Error Has Occurred")
                return
            
            datapoints = read_data(input_data)
            N = len(datapoints)

            # Perform KMeans clustering
            kmeans_labels = k_means(datapoints, K)

            # Perform SymNMF clustering
            symnmf_labels = symnmf_clusters(datapoints, N, K)
            

        else:
            print("An Error Has Occurred")
            return

    except ValueError:
        print("An Error Has Occurred")
        return

    # Calculate silhouette scores
    symnmf_silhouette = silhouette_score(datapoints, symnmf_labels)
    kmeans_silhouette = silhouette_score(datapoints, kmeans_labels)
    
    print(f"nmf: {symnmf_silhouette:.4f}")
    print(f"kmeans: {kmeans_silhouette:.4f}")

if __name__ == "__main__":
    main()
