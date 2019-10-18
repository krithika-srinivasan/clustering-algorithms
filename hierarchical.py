import sys
import math
import collections
import logging
from argparse import ArgumentParser
from util import import_file

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)

def setup_argparser():
    parser = ArgumentParser()
    parser.add_argument("--file", help="File to read data from. Format of the file should be as follows: <index> <ground truth cluster> <point 1> <point 2> ... <point n>", type=str)
    parser.add_argument("--num-clusters", type=int, help="Number of clusters", default=3)
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
        """
        We measure the distance b/w all the points in self,
        and y (Euclidean distance), and then pick the smallest distance
        from those.
        The two closest points b/w the two clusters represent the
        inter-cluster distance
        """
        dist = sys.maxsize
        for xp in self.points:
            for yp in y.points:
                dist = min(dist, xp.distance(yp))
        return dist

    def combine(self, y: 'Cluster'):
        self.points.extend(y.points)
        return self

class MinQueue:
    def __init__(self, clusters: 'List[Cluster]'):
        self.q = self._build_min_queue(clusters)

    def __len__(self):
        return len(self.q)

    def _build_min_queue(self, clusters: 'List[Cluster]'):
        q_map = collections.defaultdict(list)
        for oc in clusters:
            for ic in clusters:
                if oc != ic:
                    dist = oc.distance(ic)
                    q_map[oc].append((ic, dist))
        for cluster, dist_pairs in q_map.items():
            dist_pairs = list(set(dist_pairs))
            sorted_dist_pairs = sorted(dist_pairs, key=lambda x: x[1])
            q_map[cluster] = sorted_dist_pairs
        return q_map

    def build_min_queue(self, clusters: 'List[Cluster]'):
        self.q = self._build_min_queue(clusters)
        return self.q

    def show_min(self):
        min_dist = sys.maxsize
        min_cluster_pair = (None, None)
        for cl, dist_pairs in self.q.items():
            min_cl, dist = dist_pairs[0]
            if dist <= min_dist:
                min_dist = dist
                min_cluster_pair = (cl, min_cl)
        return (*min_cluster_pair, min_dist)

    def combine(self, x: 'Cluster', y: 'Cluster'):
        if x not in self.q or y not in self.q:
            raise ValueError('Either one of the two clusters are not in the MinQueue')
        # First remove both x and y from the MinQueue
        for cl in [x, y]:
            dist_pairs = self.q.pop(cl)
            for (ocl, dist) in dist_pairs:
                o_dist_pairs = self.q[ocl]
                # Distance is symmetric - so remove the cluster to be combined
                o_dist_pairs.remove((cl, dist))
                self.q[ocl] = sorted(o_dist_pairs, key=lambda x: x[1])
        # Combine both clusters
        x.combine(y)
        new_dists = []
        for cl, dist_pairs in self.q.items():
            dist = cl.distance(x)
            dist_pairs.append((x, dist))
            self.q[cl] = sorted(dist_pairs, key=lambda x: x[1])
            new_dists.append((cl, dist))
        new_dists = sorted(new_dists, key=lambda x: x[1])
        self.q[x] = new_dists
        return new_dists

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
        logging.info("Building initial clusters")
        self.clusters = self._build_initial_clusters(x)
        logging.info("Building min queue")
        minq = MinQueue(self.clusters)
        logging.info("Combining closest clusters using MIN distance technique..")
        while len(minq) > self.num_clusters:
            x, y, dist = minq.show_min()
            minq.combine(x, y)
        print(minq.q)

def main():
    args = setup_argparser().parse_args()

    filepath = args.file
    num_clusters = args.num_clusters

    data, truth_clusters = import_file(filepath, correct_clusters=False)
    data = [Point(x) for x in data]

    aggclustering = AgglomerativeClustering(num_clusters=num_clusters)
    aggclustering.fit(data)
    return

if __name__ == "__main__":
    main()
