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
    New = "New"
    Visited = "Visited"
    Noise = "Noise"

def query_region(data, point, eps):
    neighbours = []
    for pn in range(0, len(data)):
        if np.linalg.norm(data[point] - data[pn]) < eps:
            neighbours.append(pn)
    return neighbours

def expand_cluster(data, labels, point, neighbours, cluster_label, min_pts, eps):
    labels[point] = cluster_label
    idx = 0
    while idx < len(neighbours):
        pn = neighbours[idx]
        if labels[pn] == Marker.Noise:
            labels[pn] = cluster_label
        elif labels[pn] == Marker.New:
            labels[pn] = cluster_label
            pn_neighbours = query_region(data, pn, eps)
            if len(pn_neighbours) >= min_pts:
                neighbours = neighbours + pn_neighbours
        idx += 1
    return

def dbscan(data, min_pts, eps):
    labels = [Marker.New] * len(data)
    curr_cluster_idx = 0
    for point in range(0, len(data)):
        if labels[point] != Marker.New:
            continue
        neighbours = query_region(data, point, eps)
        if len(neighbours) < min_pts:
            labels[point] = Marker.Noise
        else:
            curr_cluster_idx += 1
            expand_cluster(data, labels, point, neighbours, curr_cluster_idx, eps, min_pts)
    return labels

def main():
    args = setup_argparser().parse_args()
    filepath = args.file
    min_pts = args.min_points
    tolerance = args.tolerance

    logging.info(args)

    data, truth_clusters = import_file(filepath)
    num_clusters = len(set(truth_clusters))

    points = np.random.rand(10) * 10
    print(points)
    labels = dbscan(points, min_pts, tolerance)
    print(labels)

    return

if __name__ == "__main__":
    main()
