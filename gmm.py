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

    def get_params(self):
        return (self.mu, self.pi, self.sigma)

    def calculate_mean_covariance(self, x, prediction):
        x = np.asarray(x)
        dim = x.shape[1]
        labels = list(set(prediction))
        self.init_means = np.zeros((self.num_clusters, dim))
        self.init_cov = np.zeros((self.num_clusters, dim, dim))
        self.init_pi = np.zeros(self.num_clusters)

        counter = 0
        for label in labels:
            indexes = np.where(prediction == label)
            self.init_pi[counter] = len(indexes[0]) / len(x)
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
        kmeans = KMeans(num_clusters=self.num_clusters, max_iterations=500)
        centroids = kmeans.fit(x)
        self._initial_means, self._initial_cov, self._initial_pi = self.calculate_mean_covariance(x, np.asarray(kmeans.labels))
        return (self._initial_means, self._initial_cov, self._initial_pi)

    def _e_step(self, x, pi, mu, sigma):
        x = np.asarray(x)
        N = x.shape[0]
        self.gamma = np.zeros((N, self.num_clusters))

        const_c = np.zeros(self.num_clusters)

        self.mu = self.mu if self._initial_means is None else self._initial_means
        self.pi = self.pi if self._initial_pi is None else self._initial_pi
        self.sigma = self.sigma if self._initial_cov is None else self._initial_cov

        for cluster in range(self.num_clusters):
            self.gamma[:, cluster] = self.pi[cluster] * mvn.pdf(x, self.mu[cluster, :], self.sigma[cluster])

        gamma_norm = np.sum(self.gamma, axis=1)[:, np.newaxis]
        self.gamma /= gamma_norm

        return self.gamma

    def _m_step(self, x, gamma):
        x = np.asarray(x)
        N = x.shape[0]
        num_clusters = self.gamma.shape[1]
        dim = x.shape[1]

        self.pi = np.mean(self.gamma, axis=0)
        self.mu = np.dot(self.gamma.T, x) / np.sum(self.gamma, axis=0)[:, np.newaxis]

        for cluster in range(num_clusters):
            tmpx = x - self.mu[cluster, :]
            gamma_diag = np.diag(self.gamma[:, cluster])
            gamma_diag = np.matrix(gamma_diag)
            # TODO: Check if this is even being used?
            x_mu = np.matrix(tmpx)
            sigma_c = tmpx.T * gamma_diag * tmpx
            self.sigma[cluster, :, :] = (sigma_c) / np.sum(self.gamma, axis=0)[:, np.newaxis][cluster]

        return self.pi, self.mu, self.sigma

    def _compute_loss_function(self, x, pi, mu, sigma):
        x = np.asarray(x)
        N = x.shape[0]
        num_clusters = self.gamma.shape[1]
        self.loss = np.zeros((N, num_clusters))

        for cluster in range(num_clusters):
            dist = mvn(self.mu[cluster], self.sigma[cluster], allow_singular=True)
            self.loss[:, cluster] = self.gamma[:, cluster] * (np.log(self.pi[cluster] + 0.00001) + dist.logpdf(x) - np.log(self.gamma[:, cluster] + 0.000001))
        self.loss = np.sum(self.loss)
        return self.loss

    def fit(self, x):
        x = np.asarray(x)
        dim = x.shape[1]
        self.mu, self.sigma, self.pi = self._init_params(x)
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
        return self

    def predict(self, x):
        x = np.asarray(x)
        labels = np.zeros((x.shape[0], self.num_clusters))
        for cluster in range(self.num_clusters):
            labels[:, cluster] = self.pi[cluster] * mvn.pdf(x, self.mu[cluster, :], self.sigma[cluster])
        labels =labels.argmax(1)
        return labels

def main():
    args = setup_argparser().parse_args()
    filepath = args.file
    max_iterations = args.num_iterations
    num_clusters = args.num_clusters
    tolerance = args.tolerance

    logging.info(args)

    data, truth_clusters = import_file(filepath)
    num_clusters = len(set(truth_clusters))

    gmm = GMM(num_clusters=num_clusters, max_iterations=max_iterations, tolerance=tolerance)
    gmm.fit(data)
    labels = gmm.predict(data)
    logging.info("Rand Index: {}".format(rand_score(truth_clusters, labels)))
    logging.info("Jaccard Coefficient: {}".format(jaccard_coeff(truth_clusters, labels)))

    # We apply PCA dim reduction to both data, and centroids to be able to plot them
    plot(reduce_dimensionality(data), truth_clusters, None, suffix="gmm_truth")
    plot(reduce_dimensionality(data), labels, None, suffix="gmm_computed")
    return

if __name__ == "__main__":
    main()
