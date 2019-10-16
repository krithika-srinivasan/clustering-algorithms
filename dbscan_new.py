import math
import logging
import numpy as np
from sklearn.cluster import DBSCAN as SKDBSCAN
from argparse import ArgumentParser
from util import import_file, rand_score, jaccard_coeff, reduce_dimensionality, plot

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)

def setup_argparser():
    parser = ArgumentParser()
    parser.add_argument("--file", help="File to read data from. Format of the file should be as follows: <index> <ground truth cluster> <point 1> <point 2> ... <point n>", type=str)
    parser.add_argument("--eps", help="Max distance cut-off", type=float, default=0.1)
    parser.add_argument("--min-points", type=int, help="Min pts", default=3)
    return parser

class Cluster:
    def __init__(self, name):
        self.name = name
        self.points = []
        self.data = None
        return

    def add_point(self, point):
        self.points.append(point)
        return

    def get_points(self):
        return self.points

    def clear_points(self):
        self.points = []
        return

    def get_dim(self, dim=0):
        return [r[dim] for r in self.points]

    def has(self, point):
        return point in self.points

    def __repr__(self):
        return "<Cluster name={0}, points={1}>".format(self.name, len(self.points))

class DBSCAN:
    def __init__(self, eps, min_points):
        self.eps = eps
        self.min_points = min_points
        self.clusters = set()
        self.visited = []
        self.labels = []
        return

    def reset_params(self):
        self.clusters = set()
        self.visited = []
        self.labels = []
        return

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

    def expand_cluster(self, cluster, point, neighbours, neighbour_indexes):
        indexes = []
        cluster.add_point(point)
        for pn_idx, pn in enumerate(neighbours):
            if pn not in self.visited:
                self.visited.append(pn)
                pn_neighbours, pn_neighbour_indexes = self.query_region(pn)
                if len(pn_neighbours) >= self.min_points:
                    for nnp_idx, nnp in enumerate(pn_neighbours):
                        if nnp not in neighbours:
                            neighbours.append(nnp)
                            neighbour_indexes.append(pn_neighbour_indexes[nnp_idx])
                for other_cluster in self.clusters:
                    if not other_cluster.has(pn):
                        if not cluster.has(pn):
                            cluster.add_point(pn)
                            indexes.append(neighbour_indexes[pn_idx])
                if len(self.clusters) == 0:
                    if not cluster.has(pn):
                        cluster.add_point(pn)
                        indexes.append(neighbour_indexes[pn_idx])
        self.clusters.add(cluster)
        return indexes

    def query_region(self, point):
        result = []
        indexes = []
        for didx, dpoint in enumerate(self.data):
            if dpoint != point:
                if self.l2_distance(dpoint, point) <= self.eps:
                    result.append(dpoint)
                    indexes.append(didx)
        return result, indexes

    def dbscan(self, data):
        self.reset_params()
        self.data = data
        self.labels = [-1] * len(data)

        noise_cluster = Cluster('Noise')
        self.clusters.add(noise_cluster)

        count = 0
        for pidx, point in enumerate(data):
            if point not in self.visited:
                count += 1
                self.visited.append(point)
                neighbours, neighbour_indexes = self.query_region(point)
                if len(neighbours) < self.min_points:
                    noise_cluster.add_point(point)
                    self.labels[pidx] = -1
                else:
                    cluster_idx = len(self.clusters)
                    name = "cluster-{0}".format(cluster_idx)
                    new_cluster = Cluster(name)
                    indexes = self.expand_cluster(new_cluster, point, neighbours, neighbour_indexes)
                    self.labels[pidx] = cluster_idx
                    for added_idx in indexes:
                        self.labels[added_idx] = cluster_idx
        return

def main():
    args = setup_argparser().parse_args()
    filepath = args.file
    min_pts = args.min_points
    eps = args.eps

    logging.info(args)

    data, truth_clusters = import_file(filepath, correct_clusters=False)

    db = DBSCAN(eps=eps, min_points=min_pts)
    db.dbscan(data)
    print(db.clusters)
    print(db.labels)
    skdb = SKDBSCAN(eps=eps, min_samples=min_pts, algorithm="brute").fit(data)
    print(np.unique(db.labels, return_counts=True))
    print(np.unique(skdb.labels_, return_counts=True))
    # plot(reduce_dimensionality(data), labels, None, suffix="dbscan")
    return

if __name__ == "__main__":
    main()
