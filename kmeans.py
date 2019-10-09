import logging
import math
import collections
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)

class KMeans:
    def __init__(self, num_clusters=3, tolerance=0.0001, max_iterations=1000):
        self.num_clusters = num_clusters
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.classes = collections.defaultdict(list)
        self.centroids = []

    def distance(self, x, y):
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

        self.centroids = [None] * self.num_clusters
        # Select first `num_clusters` data points as initial centroids
        for idx in range(self.num_clusters):
            self.centroids[idx] = x[idx]

        for epoch in range(self.max_iterations):
            is_optimal = False

            self.classes = collections.defaultdict(list)

            # Assign all points to classes
            for point in x:
                # Compute distances to all centroids
                distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
                # Assign to class that is closest
                assigned_class = distances.index(min(distances))
                self.classes[assigned_class].append(point)

            # Store current centroids
            old_centroids = self.centroids

            # Compute new centroids
            for klass, assigned_points in self.classes.items():
                self.centroids[klass] = np.mean(assigned_points, axis=0)

            for klass, current_centroid in enumerate(self.centroids):
                original_centroid = old_centroids[klass]
                if np.sum((current_centroid - original_centroid) / current_centroid * 100.0) <= self.tolerance:
                    is_optimal = True
                if is_optimal:
                    break

        return self.centroids

def main():
    kmeans = KMeans(num_clusters=3, tolerance=0.0001, max_iterations=1000)
    x = np.random.randn(10, 4)
    centroids = kmeans.fit(x)
    logging.info(x)
    logging.info(centroids)
    return

if __name__ == "__main__":
    main()
