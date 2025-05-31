import matplotlib.pyplot as plt
import sys
import os
import time
import numpy as np
from gmmx import GaussianMixtureSKLearn
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import corner
import joblib
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp

#mem = joblib.Memory('.', verbose=False)

#@mem.cache
def compute_log_likelihood(gmm, X):
    log_probs = gmm.score_samples(X)
    return np.mean(log_probs)

#@mem.cache
def fit_gmm(X_train, n_components, covariance_type='full', random_state=42, **kwargs):
    print(f"Fitting GMM with {n_components} components...")
    gmm = GaussianMixtureSKLearn(n_components=n_components, covariance_type=covariance_type, random_state=random_state, **kwargs)
    t0 = time.time()
    gmm.fit(X_train)
    return time.time() - t0, gmm

def plot_quality(gmm, X, outfilename):
    if os.environ.get('PLOT', '1') == '0':
        return
    print("plotting...")
    Y, _ = gmm.sample(100000)
    fig, axs = plt.subplots(X.shape[1], 1, figsize=(6, 20))
    for i in range(X.shape[1] - 1):
        lo = np.min(X[:,i] - X[:,i+1])
        hi = np.max(X[:,i] - X[:,i+1])
        bins = np.arange(lo, hi+0.05, 0.06)
        axs[i].hist(X[:,i] - X[:,i+1], bins=bins, color='k', histtype='step', density=True)
        axs[i].hist(Y[:,i] - Y[:,i+1], bins=bins, color='r', histtype='step', density=True)
    plt.savefig(outfilename)
    plt.close()
    print(f"plotted to {outfilename}")
    ranges = []
    for i in range(X.shape[1]):
        lo = np.nanquantile(X[:,i], 0.01)
        hi = np.nanquantile(X[:,i], 0.99)
        ranges.append((lo, hi))
    fig = corner.corner(
        X,
        range=ranges,
        color='k',
        plot_datapoints=False, plot_density=False,
        #quantiles=[0.16, 0.5, 0.84],
        title_kwargs={"fontsize": 12},
        levels=(0.64, 0.9, 0.99,),
        bins=40,
    )
    corner.corner(
        Y,
        color='gray',
        fig=fig,  # Use the same figure
        range=ranges,
        plot_datapoints=False, plot_density=False,
        fill=False,
        levels=(0.64, 0.9, 0.99,),
        bins=40
    )
    axes = np.array(fig.axes).reshape((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        axes[i,i].hist(
            X[:, i],
            bins=np.linspace(ranges[i][0], ranges[i][1], 20),
            label='obs', color='k', histtype='step', density=True)
        axes[i,i].hist(
            Y[:, i],
            bins=np.linspace(ranges[i][0], ranges[i][1], 20),
            label='GMM', color='red', histtype='step', density=True)
        axes[i,i].set_xlim(ranges[i][0], ranges[i][1])
        # axes[i,i].set_ylim(None, None)
    axes[i,i-1].legend(title="black:simulations\ngray:GMM samples")
    axes[i,i].legend()
    plt.savefig(outfilename.replace('.pdf', '') + '2d.pdf')
    plt.close()
    print(f"plotted to {outfilename.replace('.pdf', '') + '2d.pdf'}")

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from scipy.linalg import cholesky
from scipy.special import logsumexp

def local_covariances(X, indices, centroids):
    covariances = np.empty((len(centroids), X.shape[1], X.shape[1]))
    for i, idx in enumerate(indices):
        neighbors = X[idx]
        try:
            cov = np.cov(neighbors, rowvar=False)
            np.linalg.inv(cov)
            cholesky(cov, lower=True)
        except np.linalg.LinAlgError:
            cov = np.diag(np.var(neighbors, axis=0))
        covariances[i] = cov
    return covariances

def make_pd(cov, eps=1e-6):
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig < eps:
        cov += (eps - min_eig + eps) * np.eye(cov.shape[0])
    return cov

def log_likelihood_gmm(test_points, centroids, covariances, weights):
    log_probs = np.zeros((len(test_points), len(centroids)))
    for i, (mu, cov, w) in enumerate(zip(centroids, covariances, weights)):
        try:
            log_probs[:, i] = multivariate_normal.logpdf(test_points, mean=mu, cov=cov) + np.log(w)
        except np.linalg.LinAlgError:
            continue  # fallback if cov is singular
    return logsumexp(log_probs, axis=1)

def refine_weights(X, means, covariances, regularize=1e-6):
    n_samples = X.shape[0]
    n_components = means.shape[0]

    # log prob of each sample under each component
    log_resp = np.empty((n_samples, n_components))
    for k in range(n_components):
        try:
            log_resp[:, k] = multivariate_normal.logpdf(X, mean=means[k], cov=covariances[k])
        except np.linalg.LinAlgError:
            log_resp[:, k] = -np.inf

    # Normalize to get responsibilities
    log_norm = logsumexp(log_resp, axis=1, keepdims=True)  # shape (n_samples, 1)
    log_resp -= log_norm  # log-responsibilities
    resp = np.exp(log_resp)  # shape (n_samples, n_components)

    # New weights = average responsibility per component
    weights = resp.sum(axis=0) / n_samples  # shape (n_components,)
    return weights

@jax.jit
def refine_weights_jax(X, means, covariances):
    def log_prob_fn(mean, cov):
        return jax.scipy.stats.multivariate_normal.logpdf(X, mean, cov)
    # Vectorize over components
    log_probs = jax.vmap(log_prob_fn, in_axes=(0, 0))(means, covariances)  # shape (n_components, n_samples)
    log_probs = log_probs.T  # shape (n_samples, n_components)

    # Log-responsibilities
    log_resp = log_probs - jax.scipy.special.logsumexp(log_probs, axis=1, keepdims=True)

    # Convert to responsibilities
    resp = jnp.exp(log_resp)

    # Compute new weights
    weights = resp.sum(axis=0) / X.shape[0]
    return weights

class LightGMM:
    def __init__(self, n_components, init_kwargs=dict(n_init=1, max_iter=2, init='random'), refine_weights=False):
        init_kwargs['n_clusters'] = n_components
        #self.n_neighbours = n_neighbours
        self.init_kwargs = init_kwargs
        #self.q = q
        self.n_components = n_components
        self.initialised = False
        self.refine_weights = refine_weights
    def _init(self, X):
        self.kmeans_ = KMeans(**self.init_kwargs).fit(X)
        self.initialised = True
        self.means_ = self.kmeans_.cluster_centers_
        labels = self.kmeans_.labels_
        weights = np.bincount(labels, minlength=self.n_components).astype(np.float64)
        weights /= weights.sum()
        self.weights_ = weights
        #if isinstance(self.n_neighbours, int):
        #    self.n_neighbours_ = self.n_neighbours
        #elif isinstance(self.n_neighbours, float):
        #    self.n_neighbours_ = int(len(X) / self.n_components * self.n_neighbours)
        #    print('n_neighbours:', self.n_neighbours_)
        #else:
        #    raise ValueError(f"n_neighbors must be int or float, not {self.n_neighbours}")
        #if isinstance(self.q, int):
        #    self.q_ = self.q
        #elif isinstance(self.q, float):
        #    self.q_ = max(1, int(X.shape[1] * self.q))
        #    print('q:', self.q_)
        #else:
        #    raise ValueError(f"q must be int or float, not {self.q}")
        #self.pca = PCA(n_components=self.q_)
        #self.nn = NearestNeighbors(n_neighbors=self.n_neighbours_, leaf_size=20)
    def fit(self, X):
        if not self.initialised:
            self._init(X)
        #t0 = time.time()
        #X_proj = self.pca.fit_transform(X)
        #mu_proj = self.pca.transform(self.means_)
        #t1 = time.time()
        #self.nn.fit(X_proj)
        #t2 = time.time()
        #_, indices = self.nn.kneighbors(mu_proj)
        #t3 = time.time()
        #print(f"{t1-t0:.2f}s PCA, {t2-t1:.2f}s nearest neighbours fit, {t3-t2:.2f}s nearest neighbours lookup")
        indices = self.kmeans_.labels_[None,:] == np.arange(self.n_components)[:,None]
        self.covariances_ = local_covariances(X, indices, self.means_)
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, 'full')
        if self.refine_weights:
            self.weights_ = refine_weights_jax(X, self.means_, self.covariances_)
            pass

        self.converged_ = True
        self.n_iter_ = 0
    def to_sklearn(self):
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            warm_start=True,
            weights_init=self.weights_,
            means_init=self.means_,
            precisions_init=self.precisions_cholesky_,
        )
        # This does a warm start at the given parameters
        gmm.converged_ = True
        #gmm.lower_bound_ = -np.inf
        gmm.weights_ = self.weights_
        gmm.means_ = self.means_
        gmm.precisions_cholesky_ = self.precisions_cholesky_
        gmm.covariances_ = self.covariances_
        return gmm
    def score_samples(self, X):
        return log_likelihood_gmm(X, self.means_, self.covariances_, self.weights_)

def main():
    if len(sys.argv) < 2:
        print("Usage: python gmm_compare.py data.txt.gz")
        return

    filename = sys.argv[1]
    covariance_type = sys.argv[2]
    X = np.loadtxt(filename)
    print(f"Loaded data with shape: {X.shape}")

    X_train, X_test = train_test_split(X, test_size=0.5, random_state=42)
    print(f"Training data: {X_train.shape} Test data: {X_test.shape}")

    for n_components in 5, 10, 15, 20, 30, 40, 80, 120, 200:
        # weights = np.ones(n_components) / n_components
        #centroids = X_train[np.random.choice(len(X_train), n_components, replace=False)]
        for i in range(2):   # two runs to allow jax.jit to warm up.
            gmm = LightGMM(n_components) #, refine_weights=True)
            t0 = time.time()
            gmm._init(X_train)
            gmm.fit(X_train)
            dt = time.time() - t0
        print(f'{dt:.2f}s [Kmeans+cov]')
        #ll_fast = log_likelihood_gmm(X_test, centroids, covariances, weights)
        #ll_fast = compute_log_likelihood(gmm, X_test)
        #print(f"{dt:.2f}s [protoGMM] K={n_components} Test log-likelihood: {ll_fast:.4f}")
        gmmsk = gmm.to_sklearn()
        ll_fast = compute_log_likelihood(gmmsk, X_test)
        print(f"{dt:.2f}s [protoGMM] K={n_components} Test log-likelihood: {ll_fast:.4f}")
        #plot_quality(gmmsk, X_test, f'{filename}_gmmlight_{n_components}.pdf')

    for n_components in 10, 20, 40, 80:
        dt, global_gmm = fit_gmm(X_train, n_components=n_components, n_init=1, max_iter=1, covariance_type=covariance_type)
        ll_global = compute_log_likelihood(global_gmm, X_test)
        #plot_quality(global_gmm, X_test, f'{filename}_gmm_global{covariance_type}_{n_components}.pdf')
        print(f"{dt:.2f}s [Global GMM] K={n_components} Test log-likelihood: {ll_global:.4f}")


if __name__ == "__main__":
    main()
