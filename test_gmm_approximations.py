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
    print(f"Fitting BGMM with {n_components} components...")
    gmm = BayesianGaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state, **kwargs)
    gmm.fit(X_train)
    return gmm

@mem.cache
def fit_stratified_gmm(X_train, X_test, n_bins=8, local_components=5, covariance_type='full'):
    print(f"Fitting stratified GMM: {n_bins} bins × {local_components} components each...")
    
    pca = PCA(n_components=1)
    pc1_train = pca.fit_transform(X_train).flatten()
    # pc1_test = pca.transform(X_test).flatten()

    bin_edges = np.linspace(pc1_train.min(), pc1_train.max(), n_bins + 1)
    bin_indices = np.digitize(pc1_train, bin_edges) - 1

    all_weights = []
    all_means = []
    all_covariances = []
    all_precisions = []
    all_precisions_cholesky = []

    for i in range(n_bins):
        mask = bin_indices == i
        print(f"  bin {i}: {mask.sum()} members")
        X_bin = X_train[mask]
        if len(X_bin) < local_components:
            continue  # skip if too few points

        gmm_bin = fit_gmm(X_bin, n_components=local_components, covariance_type=covariance_type, random_state=42, n_init=10)

        weight = len(X_bin) / len(X_train)
        all_weights.extend(weight * gmm_bin.weights_)
        all_means.extend(gmm_bin.means_)
        all_covariances.extend(gmm_bin.covariances_)
        all_precisions.extend(gmm_bin.precisions_)
        all_precisions_cholesky.extend(gmm_bin.precisions_cholesky_)

    print(f"Total combined components: {len(all_weights)}")
    combined_gmm = GaussianMixture(
        n_components=len(all_weights),
        warm_start=True,
        covariance_type='full',
        weights_init=np.array(all_weights) / np.sum(all_weights),
        means_init = np.array(all_means),
        precisions_init = np.array(all_precisions),
    )
    combined_gmm.weights_ = np.array(all_weights) / np.sum(all_weights)
    combined_gmm.means_ = np.array(all_means)
    combined_gmm.covariances_ = np.array(all_covariances)
    # sklearn expects precisions to be computed
    combined_gmm.precisions_ = np.array(all_precisions)
    combined_gmm.precisions_cholesky_ = np.array(all_precisions_cholesky)

    return combined_gmm

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

    filename = sys.argv[1]
    covariance_type = sys.argv[2]
    X = np.loadtxt(filename)
    print(f"Loaded data with shape: {X.shape}")

    X_train, X_test = train_test_split(X, test_size=0.9, random_state=42)
    print(f"Training data: {X_train.shape} Test data: {X_test.shape}")

    for n_components in 10, 15, 20, 30, 40, 80:
        global_gmm = fit_gmm(X_train, n_components=n_components, n_init=10, covariance_type=covariance_type)
        ll_global = compute_log_likelihood(global_gmm, X_test)
        plot_quality(global_gmm, X_test, f'{filename}_gmm_global{covariance_type}_{n_components}.pdf')
        print(f"[Global GMM] K={n_components} Test log-likelihood: {ll_global:.4f} it={global_gmm.n_iter_}")

    # Global GMM
    for n_components in 10, 15, 20, 30, 40, 80:
        break
        global_gmm = fit_bgmm(X_train, n_components=n_components, n_init=10, covariance_type=covariance_type)
        ll_global = compute_log_likelihood(global_gmm, X_test)
        plot_quality(global_gmm, X_test, f'{filename}_gmm_global{covariance_type}_{n_components}.pdf')
        print(f"[Global BGMM] K={n_components} Test log-likelihood: {ll_global:.4f} it={global_gmm.n_iter_}")

    # Stratified GMM
    for n_bins in 10, 8, 6, 4, 2:
        for local_components in 1, 2, 4, 6:
            break
            stratified_gmm = fit_stratified_gmm(X_train, X_test, n_bins=n_bins, local_components=local_components)
            ll_stratified = compute_log_likelihood(stratified_gmm, X_test)
            print(f"[Stratified GMM] n_bins={n_bins} K={local_components} Test log-likelihood: {ll_stratified:.4f} it={global_gmm.n_iter_}")
            plot_quality(stratified_gmm, X_test, f'{filename}_gmm_strat{covariance_type}{n_bins}_{local_components}.pdf')
            stratified_gmm.fit(X_train)
            ll_postprocessed = compute_log_likelihood(stratified_gmm, X_test)
            print(f"[Postprocessed GMM] n_bins={n_bins} K={stratified_gmm.n_components} Test log-likelihood: {ll_postprocessed:.4f} it={global_gmm.n_iter_}")
            plot_quality(stratified_gmm, X_test, f'{filename}_gmm_strat{covariance_type}{n_bins}P_{local_components}.pdf')

if __name__ == "__main__":
    main()
