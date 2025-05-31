import matplotlib.pyplot as plt
import sys
import os
import time
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import corner
import joblib

mem = joblib.Memory('.', verbose=False)

@mem.cache
def compute_log_likelihood(gmm, X):
    log_probs = gmm.score_samples(X)
    return np.mean(log_probs)

@mem.cache
def fit_gmm(X_train, n_components, covariance_type='full', random_state=42, **kwargs):
    t0 = time.time()
    print(f"Fitting GMM with {n_components} components...")
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state, **kwargs)
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
    print(f"plotted to {outfilename.replace('.pdf', '') + '_2d.pdf'}")

from Mixture_Models import Mclust, MFA

@mem.cache
def fit_Mclust(data, data_test, constraint, K, scale=0.5, maxiter=100, **fit_kwargs):
    t0 = time.time()
    test_Mclust = Mclust(data, constraint=constraint)
    init_params = test_Mclust.init_params(num_components=K, scale=scale)
    print('init:', init_params)
    params_store = test_Mclust.fit(
        init_params, **fit_kwargs
    )
    test_Mclust.data = data_test
    return time.time() - t0, test_Mclust.likelihood(params_store[-1])

def make_pd(cov, eps=1e-6):
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig < eps:
        cov += (eps - min_eig + eps) * np.eye(cov.shape[0])
    return cov

def MFA_to_GMM(final_params):
    log_pi = final_params["log proportions"]
    pi = np.exp(log_pi - log_pi.max())  # Convert to unnormalized weights
    pi /= pi.sum()                      # Normalize to sum to 1

    means = final_params["means"]               # shape: (K, d)
    A = final_params["fac_loadings"]            # shape: (K, d, q)
    error_diag = final_params["error"]          # shape: (K, d)

    # Compute full covariances
    covariances = np.array([
        A_k @ A_k.T + np.diag(error_k)
        for A_k, error_k in zip(A, error_diag)
    ])  # shape: (K, d, d)
    
    gmm = GaussianMixture(n_components=len(pi), covariance_type='full')
    gmm.weights_ = pi
    gmm.means_ = means
    gmm.covariances_ = covariances
    from scipy.linalg import cho_factor
    gmm.precisions_cholesky_ = np.array([
        cho_factor(np.linalg.inv(make_pd(cov)))[0]
        for cov in covariances
    ])
    #gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))
    return gmm

@mem.cache
def fit_MFA(data, data_test, K, q, scale=1, maxiter=100, **fit_kwargs):
    t0 = time.time()
    test_MFA = MFA(data)
    init_params = test_MFA.init_params(num_components=K, q=q, scale=scale)
    print('init:', init_params)
    params_store = test_MFA.fit(
        init_params, **fit_kwargs
    )
    test_MFA.data = data_test
    return time.time() - t0, MFA_to_GMM(params_store[-1]), test_MFA.likelihood(params_store[-1])

def main():
    if len(sys.argv) < 2:
        print("Usage: python gmm_compare.py data.txt.gz covariance_type")
        return

    filename = sys.argv[1]
    covariance_type = sys.argv[2]
    X = np.loadtxt(filename)
    print(f"Loaded data with shape: {X.shape}")

    assert np.isfinite(X).all()
    X_train, X_test = train_test_split(X, test_size=0.95, random_state=42)
    print(f"Training data: {X_train.shape} Test data: {X_test.shape}")

    # Global GMM
    for n_components in 10, 20, 40, 80:
        dt, global_gmm = fit_gmm(X_train, n_components=n_components, n_init=1, covariance_type=covariance_type)
        ll_global = compute_log_likelihood(global_gmm, X_test)
        plot_quality(global_gmm, X_test, f'{filename}_gmm_global{covariance_type}_{n_components}.pdf')
        print(f"[Global GMM] K={n_components} Test log-likelihood: {ll_global:.4f} it={global_gmm.n_iter_}")
    
    """
    for n_components in 2, 4, 8:
        for constraint in ['EVI', 'VEI', 'VVI', 'EVE', 'EII', 'VII', 'VVE', 'EEE', 'EEI', 'EEV', 'VEV', 'VVV', 'EVV', 'VEE']:
            dt, like_test = fit_Mclust(X_train, X_test, constraint=constraint, K=n_components, scale=1,
                opt_routine="grad_descent", learning_rate=0.0005, mass=0.9, maxiter=100)
            print(f'Mclust {constraint} K={n_components}: {like_test:.3f}')
    """
    for n_components in 2, 4, 8, 12, 20:
        for q in 6, 4, 2, 1:
            np.random.seed(42)
            try:
                dt, gmm, like_test = fit_MFA(X_train, X_test, K=n_components, q=q, scale=1,
                    opt_routine="grad_descent", learning_rate=0.0005, mass=0.9, maxiter=10)
                like_test2 = compute_log_likelihood(gmm, X_test)
                print(f'Mclust q={q} K={n_components}: {like_test:.3f} asGMM:{like_test2:.3f}')
            except np.linalg.LinAlgError:
                pass


if __name__ == "__main__":
    main()
