import math
import logging
import numpy as np
from argparse import ArgumentParser
from util import import_file, rand_score, jaccard_coeff, reduce_dimensionality, plot

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)

def setup_argparser():
    parser = ArgumentParser()
    parser.add_argument("--file", help="File to read data from. Format of the file should be as follows: <index> <ground truth cluster> <point 1> <point 2> ... <point n>", type=str)
    parser.add_argument("--tolerance", help="Tolerance for centroid shifts", type=float, default=0.1)
    parser.add_argument("--min-points", type=int, help="Min pts", default=3)
    return parser

class Marker:
    Unvisited = 0
    Noise = -1

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

def query_region(data, point, eps):
    neighbours = []
    for pn in range(0, len(data)):
        if np.linalg.norm(l2_distance(data[point], data[pn])) < eps:
            neighbours.append(pn)
    return neighbours

def expand_cluster(data, labels, point, neighbours, cluster_label, min_pts, eps):
    labels[point] = cluster_label
    idx = 0
    while idx < len(neighbours):
        pn = neighbours[idx]
        if labels[pn] == Marker.Noise:
            labels[pn] = cluster_label
        elif labels[pn] == Marker.Unvisited:
            labels[pn] = cluster_label
            pn_neighbours = query_region(data, pn, eps)
            if len(pn_neighbours) >= min_pts:
                neighbours = neighbours + pn_neighbours
        idx += 1
    return

def dbscan(data, min_pts, eps):
    labels = [Marker.Unvisited] * len(data)
    curr_cluster_idx = 0
    for idx, point in enumerate(data):
        if labels[idx] != Marker.Unvisited:
            continue
        neighbours = query_region(data, idx, eps)
        if len(neighbours) < min_pts:
            labels[idx] = Marker.Noise
        else:
            curr_cluster_idx += 1
            expand_cluster(data, labels, idx, neighbours, curr_cluster_idx, eps, min_pts)
    return labels

def main():
    args = setup_argparser().parse_args()
    filepath = args.file
    min_pts = args.min_points
    tolerance = args.tolerance

    logging.info(args)

    data, truth_clusters = import_file(filepath, correct_clusters=False)

    labels = dbscan(data, min_pts, tolerance)
    logging.info("Rand index: {}".format(rand_score(truth_clusters, labels)))
    logging.info("Jaccard coeff: {}".format(jaccard_coeff(truth_clusters, labels)))

    plot(reduce_dimensionality(data), labels, None, suffix="dbscan")
    return

if __name__ == "__main__":
    main()
