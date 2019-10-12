import sys
import math
import logging
from argparse import ArgumentParser
from util import import_file

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)

def setup_argparser():
    parser = ArgumentParser()
    parser.add_argument("--file", help="File to read data from. Format of the file should be as follows: <index> <ground truth cluster> <point 1> <point 2> ... <point n>", type=str)
    return parser

def l2_distance(x, y):
    """
    Euclidean distance b/w two points.
    """
    if len(x) != len(y):
        raise ValueError("Dimensions of x {} and y {} are not the same".format(len(x), len(y)))
    dist = 0.0
    for idx in range(len(x)):
        dist += (x[idx] - y[idx]) ** 2
    return math.sqrt(dist)

class Point:
    def __init__(self, data: 'List[float]'):
        self.data = data
        return

    def __repr__(self):
        return "<Point ({0})>".format(",".join([str(round(x, 1)) for x in self.data]))

    def distance(self, y: 'Point'):
        return l2_distance(self.data, y.data)

class Cluster:
    def __init__(self, points):
        self.points = points
        return

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return "<Cluster id={0} num_points={1}>".format(id(self), len(self))

    def distance(self, y: 'Cluster'):
        dist = sys.maxsize
        for xp in self.points:
            for yp in y.points:
                dist = min(dist, xp.distance(yp))
        return dist

class MinQueue:
    def __init__(self, clusters: 'List[Cluster]'):
        self.q = self._build_min_queue(clusters)

    def _build_min_queue(self, clusters: 'List[Cluster]'):
        q_map = {}
        for oidx, oc in enumerate(clusters):
            min_dist = sys.maxsize
            min_idx = -1
            for idx, ic in enumerate(clusters):
                if oc != ic:
                    dist = oc.distance(ic)
                    if dist <= min_dist:
                        min_dist = dist
                        min_idx = idx
            key = (oidx, min_idx)
            q_map[key] = min_dist
        q = []
        for (fidx, sidx), dist in q_map.items():
            q.append((clusters[fidx], clusters[sidx], dist))
        # Sort by distance between clusters
        q = sorted(q, key=lambda x: x[2])
        return q

    def build_min_queue(self, clusters: 'List[Cluster]'):
        self.q = self._build_min_queue(clusters)
        return self.q

class AgglomerativeClustering:
    def __init__(self, num_clusters=3):
        self.num_clusters = num_clusters
        self.clusters = []
        return

    def _build_initial_clusters(self, x: 'List[Point]'):
        clusters = []
        for xp in x:
            clusters.append(Cluster([xp]))
        return clusters

    def fit(self, x):
        # TODO: Complete the actual clustering
        old_clusters = set()
        logging.info("Building initial clusters")
        self.clusters = self._build_initial_clusters(x)
        logging.info("Building min queue")
        minq = MinQueue(self.clusters)

def main():
    args = setup_argparser().parse_args()
    data, truth_clusters = import_file(args.file)
    data = [Point(x) for x in data]

    aggclustering = AgglomerativeClustering()
    aggclustering.fit(data)
    return

if __name__ == "__main__":
    main()
