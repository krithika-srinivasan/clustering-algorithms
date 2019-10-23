# CSE 601, Project 2, Clustering Algorithms

## How to run

### KMeans, Hierarchical, GMM, DBSCAN
These 4 clustering algorithms are implemented in Python 3.

The following packages are required to run them:

1. Numpy
2. Scipy
3. Scikit-Learn
4. Matplotlib

When each of the 4 python implementations are run, they will also generate 2 image files, with a name like so `cluster_<algorithm>_computed.png` or `cluster_algorithm_truth.png`. These files are the computed, and actual ground truth cluster based (respectively), 2d cluster plots.

You can use variations of the following commands to run each algorithm.

KMeans
```python
python3 kmeans.py --file ./data/cho.txt --tolerance 0.0001 --num-iterations 1000 --num-clusters 5
```

Hierarchical
```python
python3 hierarchical.py --file ./data/cho.txt --num-clusters 5
```

GMM
```python
python3 gmm.py --file ./data/cho.txt --num-iterations 1000 --num-clusters 5 --tolerance 0.0001
```

DBSCAN
```python
python dbscan.py --file ./data/cho.txt --min-points 3 --eps 1
```

## REMOVE EVERYTHING BELOW THIS LINE BEFORE SUBMITTING - INCLUDING THIS LINE
## TODO:
2. Add provision to accept pi, mu, sigma for GMM, output estimated parameters

