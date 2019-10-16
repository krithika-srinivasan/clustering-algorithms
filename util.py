import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, jaccard_score

def import_file(path, correct_clusters=True):
    truth_clusters, data = [], []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            idx, truth_cluster, *points = line.split()
            truth_clusters.append(int(truth_cluster))
            data.append([float(x) for x in points])
    # We have zero-indexed clusters.
    # So we need to make sure that imported data is also zero-indexed
    if correct_clusters and min(truth_clusters) > 0:
        truth_clusters = np.subtract(truth_clusters, min(truth_clusters))
    return data, truth_clusters

def rand_score(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)

def jaccard_coeff(labels_true, labels_pred):
    # Average is set to "weighted" here to take into label imbalance
    return jaccard_score(labels_true, labels_pred, average="weighted")

def plot(x, labels, centroids=None, suffix=None):
    x = np.asarray(x)
    uniq_labels = sorted(set(labels))
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    for label in uniq_labels:
        indexes = [idx for idx, l in enumerate(labels) if l == label]
        ax.scatter(x[indexes, 0], x[indexes, 1], label=label)
    # Plot the centroids
    if centroids is not None:
        centroids = np.asarray(centroids)
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="X")
    plt.title("Clustering results")
    plt.legend()
    if suffix:
        plt.savefig("cluster_{0}.png".format(suffix))
    plt.show()
    return

def reduce_dimensionality(x, dim=2):
    return PCA(n_components=dim).fit_transform(x)
