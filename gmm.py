import sys
import logging
import numpy as np
from scipy.stats import multivariate_normal as mvn
from kmeans import KMeans
from argparse import ArgumentParser
from util import import_file, rand_score, jaccard_coeff, reduce_dimensionality, plot

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)

def setup_argparser():
    parser = ArgumentParser()
    parser.add_argument("--file", help="File to read data from. Format of the file should be as follows: <index> <ground truth cluster> <point 1> <point 2> ... <point n>", type=str)
    parser.add_argument("--num-iterations", help="Max iterations to run the GMM algorithm for", type=int, default=1000)
    parser.add_argument("--num-clusters", help="Number of clusters", type=int, default=5)
    parser.add_argument("--tolerance", help="Tolerance for loss", type=float, default=0.0001)
    return parser

class GMM:
    def __init__(self, num_clusters, max_iterations=1000, tolerance=1e-4):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        return

    def calculate_mean_covariance(self, x, prediction):
        """
        Calculate mean and cov for each cluster given to us
        by the KMeans clustering
        """
        x = np.asarray(x)
        dim = x.shape[1]
        labels = list(set(prediction))
        self.init_means = np.zeros((self.num_clusters, dim))
        self.init_cov = np.zeros((self.num_clusters, dim, dim))
        self.init_pi = np.zeros(self.num_clusters)

        counter = 0
        for label in labels:
            indexes = np.where(prediction == label)
            self.init_pi[counter] = len(indexes[0]) / x.shape[0]
            self.init_means[counter, :] = np.mean(x[indexes], axis=0)
            centered = x[indexes] - self.init_means[counter, :]
            num_points = x[indexes].shape[0]
            self.init_cov[counter, :, :] = np.dot(self.init_pi[counter] * centered.T, centered) / num_points
            counter += 1
        # When we use our own labels, we get a sum of 0.9999 - using Sklearns gives us a sum of 1.0
        # So, no need to assert
        # assert np.sum(self.init_pi) == 1
        return (self.init_means, self.init_cov, self.init_pi)

    def _init_params(self, x):
        """
        Use KMeans to find starting values
        """
        kmeans = KMeans(num_clusters=self.num_clusters, max_iterations=500)
        centroids = kmeans.fit(x)
        self._initial_means, self._initial_cov, self._initial_pi = self.calculate_mean_covariance(x, np.asarray(kmeans.labels))
        return (self._initial_means, self._initial_cov, self._initial_pi)

    def _e_step(self, x, pi, mu, sigma):
        """
        Perform Expectation step

        x - Data points
        pi - Weights of mixture components
        mu - Means of mixture components
        sigma: Covar matrices of mixture components
        """
        x = np.asarray(x)
        N = x.shape[0]
        self.gamma = np.zeros((N, self.num_clusters))

        const_c = np.zeros(self.num_clusters)

        self.mu = self.mu if self._initial_means is None else self._initial_means
        self.pi = self.pi if self._initial_pi is None else self._initial_pi
        self.sigma = self.sigma if self._initial_cov is None else self._initial_cov

        for cluster in range(self.num_clusters):
            # Convert covar matrix from singular matrix to regular
            self.sigma[cluster] = correct_singular_matrix(self.sigma[cluster])
            dist = mvn(self.mu[cluster, :], self.sigma[cluster], allow_singular=False)
            # Using this line - some clusters are ignored. What gives?
            self.gamma[:, cluster] = self.pi[cluster] * dist.pdf(x)
            # TODO: If we use this line below, everything clusters into one cluster
            # self.gamma[:, cluster] = self.pi[cluster] * mvn.pdf(x, self.mu[cluster, :], self.sigma[cluster])

        # Normalise gamma to give a probability
        gamma_norm = np.sum(self.gamma, axis=1)[:, np.newaxis]
        self.gamma /= gamma_norm

        return self.gamma

    def _m_step(self, x, gamma):
        """
        Perform Maximisation.
        Need to update the priors, means, and covar matrix
        """
        x = np.asarray(x)
        N, dim = x.shape[0], x.shape[1]
        num_clusters = self.gamma.shape[1]

        self.pi = np.mean(self.gamma, axis=0)
        self.mu = np.dot(self.gamma.T, x) / np.sum(self.gamma, axis=0)[:, np.newaxis]

        for cluster in range(num_clusters):
            x_centered = x - self.mu[cluster, :]
            gamma_diag = np.diag(self.gamma[:, cluster])
            gamma_diag = np.matrix(gamma_diag)
            sigma_c = x_centered.T * gamma_diag * x_centered
            self.sigma[cluster, :, :] = (sigma_c) / np.sum(self.gamma, axis=0)[:, np.newaxis][cluster]

        return self.pi, self.mu, self.sigma

    def _compute_loss_function(self, x, pi, mu, sigma):
        """
        Compute lower bound loss
        """
        x = np.asarray(x)
        N = x.shape[0]
        num_clusters = self.gamma.shape[1]
        self.loss = np.zeros((N, num_clusters))

        for cluster in range(num_clusters):
            # Convert covar matrix from singular matrix to regular
            self.sigma[cluster] = correct_singular_matrix(self.sigma[cluster])
            dist = mvn(self.mu[cluster], self.sigma[cluster], allow_singular=False)
            self.loss[:, cluster] = self.gamma[:, cluster] * (np.log(self.pi[cluster] + 0.00001) + dist.logpdf(x) - np.log(self.gamma[:, cluster] + 0.000001))

        self.loss = np.sum(self.loss)
        return self.loss

    def fit(self, x, mu=None, sigma=None, pi=None):
        x = np.asarray(x)
        dim = x.shape[1]
        if not mu or not sigma or not pi:
            logging.info("Either one of Mu, Sigma, or Pi not provided to fit with - calculating using KMeans..")
            self.mu, self.sigma, self.pi = self._init_params(x)
        else:
            logging.info("Using user-provided Mu, Sigma, Pi values")
            self.mu, self.sigma, self.pi = mu, sigma, pi
        last_loss = -1
        try:
            for run in range(self.max_iterations):
                self.gamma = self._e_step(x, self.pi, self.mu, self.sigma)
                self.pi, self.mu, self.sigma = self._m_step(x, self.gamma)
                loss = self._compute_loss_function(x, self.pi, self.mu, self.sigma)
                if run == 0:
                    last_loss = loss
                else:
                    if abs(loss - last_loss) <= self.tolerance:
                        logging.info("Iteration: {}, Loss: {}".format(run, loss))
                        break
                    else:
                        last_loss = loss
                if run % 10 == 0:
                    logging.info("Iteration: {}, Loss: {}".format(run, loss))
        except Exception as e:
            print(e)
        # Argmax along cluster axis gives us the labels
        self.labels = self.gamma.argmax(1)
        return self

def correct_singular_matrix(x, eps=1e-6):
    x = np.asarray(x)
    if np.linalg.cond(x) < 1 / sys.float_info.epsilon:
        # Not a singular matrix
        return x
    # Add a small value along the diagonal
    return x + (np.eye(x.shape[1]) * eps)

def main():
    args = setup_argparser().parse_args()
    filepath = args.file
    max_iterations = args.num_iterations
    num_clusters = args.num_clusters
    tolerance = args.tolerance

    logging.info(args)

    data, truth_clusters = import_file(filepath)
    # num_clusters = len(set(truth_clusters))

    gmm = GMM(num_clusters=num_clusters, max_iterations=max_iterations, tolerance=tolerance)
    mu = [[1, 1], [3.5, 5.3], [0, 4]]
    pi = [0.1, 0.8, 0.1]
    sigma = [[[1, 0.5], [0.5, 1]], [[1, 0], [0, 2]], [[0.5, 0], [0, 0.1]]]
    # gmm.fit(data, mu=mu, sigma=sigma, pi=pi)
    gmm.fit(data)
    labels = gmm.labels
    logging.info("Final Pi: {}, Mu: {}, Sigma: {}".format(gmm.pi, gmm.mu, gmm.sigma))
    logging.info("Rand Index: {}".format(rand_score(truth_clusters, labels)))
    logging.info("Jaccard Coefficient: {}".format(jaccard_coeff(truth_clusters, labels)))

    # We apply PCA dim reduction to both data, and centroids to be able to plot them
    plot(reduce_dimensionality(data), truth_clusters, None, suffix="gmm_truth")
    plot(reduce_dimensionality(data), labels, None, suffix="gmm_computed")
    return

if __name__ == "__main__":
    main()
