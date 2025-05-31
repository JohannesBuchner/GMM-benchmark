import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import corner
import joblib

mem = joblib.Memory('.', verbose=False)

@mem.cache
def compute_log_likelihood(gmm, X):
    log_probs = gmm.score_samples(X)
    return np.mean(log_probs)

@mem.cache
def fit_gmm(X_train, n_components, covariance_type='full', random_state=42, **kwargs):
    print(f"Fitting GMM with {n_components} components...")
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state, **kwargs)
    gmm.fit(X_train)
    return gmm

@mem.cache
def fit_bgmm(X_train, n_components, covariance_type='full', random_state=42, **kwargs):
    print(f"Fitting GMM with {n_components} components...")
    gmm = BayesianGaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state, **kwargs)
    gmm.fit(X_train)
    return gmm

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
    print(f"plotted to {outfilename.replace('.pdf', '') + '_2d.pdf'}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python gmm_compare.py data.txt.gz")
        return

    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    covariance_type = 'full'
    X1 = np.loadtxt(filename1)
    X2 = np.loadtxt(filename2)

    X1_train, X1_test = train_test_split(X1, test_size=0.95, random_state=42)
    X2_train, X2_test = train_test_split(X2, test_size=0.95, random_state=42)

    # Global GMM
    n_components = int(sys.argv[3])
    n_iter = int(sys.argv[4])
    print(f"Training data: {X1_train.shape} Test data: {X1_test.shape}")
    gmm = fit_gmm(X1_train, n_components=n_components, n_init=1, covariance_type=covariance_type, warm_start=False)
    ll = compute_log_likelihood(gmm, X1_test)
    print(f"Test 1 log-likelihood: {ll:.4f} it={gmm.n_iter_}")
    ll = compute_log_likelihood(gmm, X2_test)
    print(f"Test 2 log-likelihood: {ll:.4f} it={gmm.n_iter_}")
    plot_quality(gmm, X1_test, f'{filename1}_gmm_{n_components}_diff.pdf')
    print(f"Training data: {X2_train.shape} Test data: {X2_test.shape}")
    print(f"adapting, n_iter={n_iter}")
    gmm.max_iter = n_iter
    gmm.fit(X2_train)
    print(gmm.n_iter_)
    plot_quality(gmm, X2_test, f'{filename2}_gmm_{n_components}_diff.pdf')
    ll = compute_log_likelihood(gmm, X1_test)
    print(f"Test 1 log-likelihood: {ll:.4f} it={gmm.n_iter_}")
    ll = compute_log_likelihood(gmm, X2_test)
    print(f"Test 2 log-likelihood: {ll:.4f} it={gmm.n_iter_}")
    

if __name__ == "__main__":
    main()
