import logging
import math
import random
import collections
import numpy as np
from argparse import ArgumentParser
from util import import_file, rand_score, jaccard_coeff, reduce_dimensionality, plot

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)

def setup_argparser():
    parser = ArgumentParser()
    parser.add_argument("--file", help="File to read data from. Format of the file should be as follows: <index> <ground truth cluster> <point 1> <point 2> ... <point n>", type=str)
    parser.add_argument("--num-clusters", help="Number of clusters for KMeans", type=int, default=3)
    parser.add_argument("--random-start", help="Pick cluster centroids at random", action="store_true", default=False)
    parser.add_argument("--tolerance", help="Tolerance for centroid shifts", type=float, default=0.0001)
    parser.add_argument("--num-iterations", help="Max iterations to run the KMeans algorithm for", type=int, default=1000)
    parser.add_argument("--centroids", type=int, nargs="+", help="Initial centroid indexes - if not provided, will use first n centroids", default=None)
    return parser

def accuracy_score(predictions, actual):
    if len(predictions) != len(actual):
        raise ValueError("Dimensions of predictions and actual data are not the same")
    count = 0
    for idx, pred in enumerate(predictions):
        if pred == actual[idx]:
            count += 1
    return count / len(actual)

class KMeans:
    def __init__(self, num_clusters=3, tolerance=1e-4, max_iterations=1000):
        self.num_clusters = num_clusters
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.classes = collections.defaultdict(list)
        self.centroids = []
        self._points = None
        self.labels = []

    def l2_distance(self, x, y):
        """
        Euclidean distance b/w two points.
        """
        if len(x) != len(y):
            raise ValueError("Dimensions of x {} and y {} are not the same".format(len(x), len(y)))
        dist = 0.0
        for idx in range(len(x)):
            dist += (x[idx] - y[idx]) ** 2
        return math.sqrt(dist)

    def fit(self, x, initial_centroids=None, random_start=False):
        """
        Fits data points using the KMeans method.
        Returns optimal centroids.
        """
        self._points = x
        if initial_centroids:
            self.centroids = []
            for idx in initial_centroids:
                self.centroids.append(x[idx])
        elif random_start:
            # Select random centroids
            choices = random.sample(x, self.num_clusters)
            self.centroids = choices
        else:
            self.centroids = [None] * self.num_clusters
            # Select first `num_clusters` data points as initial centroids
            for idx in range(self.num_clusters):
                self.centroids[idx] = x[idx]

        for epoch in range(self.max_iterations):
            # Reset class/cluster/label assignments
            self.classes = collections.defaultdict(list)
            self.labels = []

            # Assign all points to classes
            for point in x:
                # TODO: Figure out how to take care of outlier case
                klass = self.predict(point)
                self.classes[klass].append(point)
                self.labels.append(klass)

            # Store current centroids
            old_centroids = [c for c in self.centroids]

            # Compute new centroids
            for klass, assigned_points in self.classes.items():
                self.centroids[klass] = np.mean(assigned_points, axis=0)

            is_optimal = [False] * self.num_clusters
            # Measure the shift between the old and new centroids
            for klass, current_centroid in enumerate(self.centroids):
                original_centroid = old_centroids[klass]
                shift = self.l2_distance(current_centroid, original_centroid)
                if shift <= self.tolerance:
                    is_optimal[klass] = True

            # If none of the centroids have moved within tolerance - done optimizing
            if all(is_optimal) == True:
                logging.info("Optimal centroids found at Epoch {}".format(epoch))
                break

        return self.centroids

    def predict(self, point):
        if not self.centroids:
            raise ValueError("No data has been fit yet!")
        # Compute distances to all centroids
        distances = [self.l2_distance(point, centroid) for centroid in self.centroids]
        # Assign to class that is closest
        klass = distances.index(min(distances))
        return klass

    def score(self, truth_clusters):
        if not self.centroids or not len(self.classes):
            raise ValueError("No data has been fit yet!")
        return accuracy_score(self.labels, truth_clusters)

def main():
    args = setup_argparser().parse_args()
    filepath = args.file
    num_clusters = args.num_clusters
    random_start = args.random_start
    max_iterations = args.num_iterations
    tolerance = args.tolerance
    initial_centroids = args.centroids

    if initial_centroids and  len(initial_centroids) != num_clusters:
        raise ValueError("Number of centroids provided does not match number of clusters")

    logging.info(args)

    data, truth_clusters = import_file(filepath, correct_clusters=True)
    # num_clusters = len(set(truth_clusters))

    kmeans = KMeans(num_clusters=num_clusters, tolerance=tolerance, max_iterations=max_iterations)
    centroids = kmeans.fit(data, initial_centroids=initial_centroids, random_start=random_start)
    score = kmeans.score(truth_clusters)

    logging.info("Centroids: {}".format(centroids))
    logging.info("Accuracy Score: {}".format(score))
    logging.info("Rand Index: {}".format(rand_score(truth_clusters, kmeans.labels)))
    logging.info("Jaccard Coefficient: {}".format(jaccard_coeff(truth_clusters, kmeans.labels)))

    # We apply PCA dim reduction to both data, and centroids to be able to plot them
    plot(reduce_dimensionality(data), truth_clusters, None, suffix="kmeans_truth")
    plot(reduce_dimensionality(data), kmeans.labels, reduce_dimensionality(centroids), suffix="kmeans_computed")
    return

if __name__ == "__main__":
    main()
