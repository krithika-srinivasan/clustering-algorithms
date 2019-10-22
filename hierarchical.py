import sys
import math
import collections
import logging
from argparse import ArgumentParser
from util import import_file, plot, reduce_dimensionality, rand_score, jaccard_coeff

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)

UNASSIGNED_LABEL = -1

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
    def __init__(self, points: 'List[Point]'):
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
        The two closest points b/w the clusters represent the
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

    def has(self, p: 'Point'):
        return p in self.points

class MinQueue:
    def __init__(self, clusters: 'List[Cluster]'):
        self.q = self._build_min_queue(clusters)

    def __len__(self):
        return len(self.q)

    def _build_min_queue(self, clusters: 'List[Cluster]'):
        """
        Build the MinQueue by calculating all the
        inter-cluster distances, and storing them as an
        adjacency list representation, sorted by distance.
        """
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

    def peek_min(self):
        """
        Return the two closest clusters currently in the MinQueue
        """
        min_dist = sys.maxsize
        min_cluster_pair = (None, None)
        for cl, dist_pairs in self.q.items():
            # Cluster, distance pairs are ordered by distances,
            # the first entry in each list is the cluster closest to the current
            min_cl, dist = dist_pairs[0]
            if dist <= min_dist:
                min_dist = dist
                min_cluster_pair = (cl, min_cl)
        return (*min_cluster_pair, min_dist)

    def combine(self, x: 'Cluster', y: 'Cluster'):
        """
        Remove both x and y from the MinQueue including
        all inter-cluster distances, combine x and y into one,
        and calculate the inter-cluster distances with this new
        cluster
        """
        if x not in self.q or y not in self.q:
            raise ValueError('Clusters to be combined are not in the MinQueue')
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
        # Compute the new inter-cluster distances, b/w the existing clusters,
        # and the new cluster.
        new_dists = []
        for cl, dist_pairs in self.q.items():
            dist = cl.distance(x)
            dist_pairs.append((x, dist))
            self.q[cl] = sorted(dist_pairs, key=lambda x: x[1])
            new_dists.append((cl, dist))
        new_dists = sorted(new_dists, key=lambda x: x[1])
        self.q[x] = new_dists
        return new_dists

    def get_clusters(self):
        return list(self.q.keys())

class AgglomerativeClustering:
    def __init__(self, num_clusters=3):
        self.num_clusters = num_clusters
        self.clusters = []
        self.data = None
        self.labels = None
        return

    def _build_initial_clusters(self, x: 'List[Point]'):
        """
        Assign each point to its own cluster
        """
        clusters = []
        for xp in x:
            clusters.append(Cluster([xp]))
        return clusters

    def fit(self, x):
        """
        Init the MinQueue, and keep combining the two closest
        clusters until we get the required number of clusters.
        """
        self.data = x
        self.labels = [UNASSIGNED_LABEL] * len(x)
        logging.info("Building initial clusters")
        self.clusters = self._build_initial_clusters(x)
        logging.info("Building min queue")
        minq = MinQueue(self.clusters)
        logging.info("Combining closest clusters using MIN distance technique..")
        while len(minq) > self.num_clusters:
            c1, c2, dist = minq.peek_min()
            minq.combine(c1, c2)
        clusters = minq.get_clusters()
        for idx, point in enumerate(x):
            for cidx, cl in enumerate(clusters):
                if cl.has(point):
                    if self.labels[idx] == UNASSIGNED_LABEL:
                        self.labels[idx] = cidx
                    else:
                        # Sound the alarm! A point is in two clusters!
                        raise ValueError('Point {0} apparently belongs to both {1} and {2}'.format(point, clusters[self.labels[idx]], cl))
        return self.labels

def main():
    args = setup_argparser().parse_args()

    filepath = args.file
    num_clusters = args.num_clusters

    data, truth_clusters = import_file(filepath, correct_clusters=True)
    points = [Point(x) for x in data]

    aggclustering = AgglomerativeClustering(num_clusters=num_clusters)
    labels = aggclustering.fit(points)
    logging.info("Labels: {}".format(labels))
    logging.info("Rand score: {}".format(rand_score(truth_clusters, labels)))
    logging.info("Jaccard coefficient: {}".format(jaccard_coeff(truth_clusters, labels)))

    # We apply PCA dim reduction to both data, and centroids to be able to plot them
    plot(reduce_dimensionality(data), truth_clusters, None, suffix="hierarchical_truth")
    plot(reduce_dimensionality(data), labels, None, suffix="hierarchical_computed")
    return

if __name__ == "__main__":
    main()
