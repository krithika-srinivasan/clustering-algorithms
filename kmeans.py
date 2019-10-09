import logging
import math
import collections
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)

def setup_argparser():
    parser = ArgumentParser()
    parser.add_argument("--file", help="File to read data from. Format of the file should be as follows: <index> <ground truth cluster> <point 1> <point 2> ... <point n>", type=str)
    return parser

def import_file(path):
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
    if min(truth_clusters) > 0:
        truth_clusters = np.subtract(truth_clusters, min(truth_clusters))
    return data, truth_clusters

class KMeans:
    def __init__(self, num_clusters=3, tolerance=0.0001, max_iterations=1000):
        self.num_clusters = num_clusters
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.classes = collections.defaultdict(list)
        self.centroids = []
        self._points = None

    def l2_distance(self, x, y):
        if len(x) != len(y):
            raise ValueError("Dimensions of x {} and y {} are not the same".format(len(x), len(y)))
        dist = 0.0
        for idx in range(len(x)):
            dist += (x[idx] - y[idx]) ** 2
        return math.sqrt(dist)

    def fit(self, x):
        """
        Fits data points using the KMeans method.
        Returns optimal centroids.
        """
        self._points = x
        self.centroids = [None] * self.num_clusters
        # Select first `num_clusters` data points as initial centroids
        for idx in range(self.num_clusters):
            self.centroids[idx] = x[idx]

        for epoch in range(self.max_iterations):
            logging.info("Epoch {}".format(epoch))
            is_optimal = [False] * self.num_clusters

            self.classes = collections.defaultdict(list)

            # Assign all points to classes
            for point in x:
                # TODO: Figure out how to take care of outlier case
                klass = self.predict(point)
                self.classes[klass].append(point)

            # Store current centroids
            old_centroids = [c for c in self.centroids]

            # Compute new centroids
            for klass, assigned_points in self.classes.items():
                self.centroids[klass] = np.mean(assigned_points, axis=0)

            for klass, current_centroid in enumerate(self.centroids):
                original_centroid = old_centroids[klass]
                shift = self.l2_distance(current_centroid, original_centroid)
                if shift <= self.tolerance:
                    is_optimal[klass] = True

            if all(is_optimal) == True:
                logging.info("Optimal centroids found")
                break

        return self.centroids

    def predict(self, point):
        if not self.centroids:
            raise ValueError("No data has been fit yet!")
        # Compute distances to all centroids
        distances = [np.linalg.norm(self.l2_distance(point, centroid)) for centroid in self.centroids]
        # Assign to class that is closest
        klass = distances.index(min(distances))
        return klass

    def score(self, truth_clusters):
        if not self.centroids or not len(self.classes):
            raise ValueError("No data has been fit yet!")
        accurate = 0
        for idx, truth in enumerate(truth_clusters):
            if self._points[idx] in self.classes[truth]:
                accurate += 1
        return (accurate / len(self._points))

def main():
    args = setup_argparser().parse_args()
    filepath = args.file
    data, truth_clusters = import_file(filepath)
    num_clusters = len(set(truth_clusters))
    logging.info("{}, {}, {}".format(num_clusters, min(truth_clusters), max(truth_clusters)))
    kmeans = KMeans(num_clusters=num_clusters, tolerance=0.00001, max_iterations=1000)
    centroids = kmeans.fit(data)
    score = kmeans.score(truth_clusters)
    logging.info("Score: {}".format(score))
    logging.info(centroids)
    return

if __name__ == "__main__":
    main()
