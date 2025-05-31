import matplotlib.pyplot as plt
import sys
import os
import time
import numpy as np
from gmmx import GaussianMixtureSKLearn
from sklearn.mixture import GaussianMixture
from askcarl.lightgmm import LightGMM
from sklearn.model_selection import train_test_split
import corner
import joblib

mem = joblib.Memory('.', verbose=False)

#@mem.cache
def compute_log_likelihood(gmm, X):
    log_probs = gmm.score_samples(X)
    return np.mean(log_probs)

@mem.cache
def fit_gmmx(X_train, n_components, covariance_type='full', random_state=42, **kwargs):
    print(f"Fitting gmmx with {n_components} components...")
    for i in range(2):
        gmm = GaussianMixtureSKLearn(n_components=n_components, covariance_type=covariance_type, random_state=random_state, **kwargs)
        t0 = time.time()
        gmm.fit(X_train)
    return time.time() - t0, gmm

@mem.cache
def fit_gmm(X_train, n_components, covariance_type='full', random_state=42, **kwargs):
    print(f"Fitting GMM with {n_components} components...")
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state, **kwargs)
    t0 = time.time()
    gmm.fit(X_train)
    return time.time() - t0, gmm

@mem.cache
def fit_lightgmm(X_train, **kwargs):
    print("Fitting LightGMM...", kwargs)
    for i in range(2):   # two runs to allow jax.jit to warm up.
        gmm = LightGMM(**kwargs)
        t0 = time.time()
        gmm.fit(X_train)
    dt = time.time() - t0
    return dt, gmm.to_sklearn()


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

def main():
    if len(sys.argv) < 2:
        print("Usage: python gmm_compare.py data.txt.gz")
        return

    filename = sys.argv[1]
    X = np.loadtxt(filename)
    print(f"Loaded data with shape: {X.shape}")
    test_size = float(sys.argv[2])

    X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
    print(f"Training data: {X_train.shape} Test data: {X_test.shape}")
    fout = open(f'{filename}_lightgmm{len(X_train)}.data', 'w')
    def pprint(s):
        print(s)
        fout.write(s)
        fout.write("\n")
        fout.flush()

    for n_components in 5, 10, 15, 20, 30, 40, 80, 120,:
        # weights = np.ones(n_components) / n_components
        #centroids = X_train[np.random.choice(len(X_train), n_components, replace=False)]
        for covariance_type in 'full',: # 'diag':
            for max_iter in 1, 100:
                for n_init in 1,: #,10
                    #dt, global_gmm = fit_gmm(X_train, n_components=n_components, max_iter=max_iter, n_init=n_init, covariance_type=covariance_type)
                    #ll_global = compute_log_likelihood(global_gmm, X_test)
                    #print(f"{dt:.2f}s {ll_global:.4f} gmm {covariance_type} K={n_components} {n_init} {max_iter}")
                    #del dt, global_gmm, ll_global

                    dt, global_gmm = fit_gmmx(X_train, n_components=n_components, max_iter=max_iter, covariance_type=covariance_type)
                    ll_global = compute_log_likelihood(global_gmm, X_test)
                    pprint(f"{dt:.2f}s {ll_global:.4f} gmmx {covariance_type} K={n_components} {n_init} {max_iter}")
                    del dt, global_gmm, ll_global

                    dt, global_gmm = fit_gmmx(X_train, n_components=n_components, max_iter=max_iter, covariance_type=covariance_type,
                        init_params='fastinit')
                    ll_global = compute_log_likelihood(global_gmm, X_test)
                    pprint(f"{dt:.2f}s {ll_global:.4f} gmmx-fastinit {covariance_type} K={n_components} {n_init} {max_iter}")
                    del dt, global_gmm, ll_global
                    continue

                    dt, global_gmm = fit_gmmx(X_train, n_components=n_components, max_iter=max_iter, covariance_type=covariance_type,
                        init_params='fastinit1')
                    ll_global = compute_log_likelihood(global_gmm, X_test)
                    pprint(f"{dt:.2f}s {ll_global:.4f} gmmx-fastinit1 {covariance_type} K={n_components} {n_init} {max_iter}")
                    del dt, global_gmm, ll_global

                    dt, global_gmm = fit_gmmx(X_train, n_components=n_components, max_iter=max_iter, covariance_type=covariance_type,
                        init_params='best-of-2')
                    ll_global = compute_log_likelihood(global_gmm, X_test)
                    pprint(f"{dt:.2f}s {ll_global:.4f} gmmx-bestof2init {covariance_type} K={n_components} {n_init} {max_iter}")
                    del dt, global_gmm, ll_global

                    dt, global_gmm = fit_gmmx(X_train, n_components=n_components, max_iter=max_iter, covariance_type=covariance_type,
                        init_params='best-of-10')
                    ll_global = compute_log_likelihood(global_gmm, X_test)
                    pprint(f"{dt:.2f}s {ll_global:.4f} gmmx-bestof10init {covariance_type} K={n_components} {n_init} {max_iter}")
                    del dt, global_gmm, ll_global

                    dt, global_gmm = fit_gmmx(X_train, n_components=n_components, max_iter=max_iter, covariance_type=covariance_type,
                        init_params='best-of-10-pc1')
                    ll_global = compute_log_likelihood(global_gmm, X_test)
                    pprint(f"{dt:.2f}s {ll_global:.4f} gmmx-bestof10pc1init {covariance_type} K={n_components} {n_init} {max_iter}")
                    del dt, global_gmm, ll_global

                    dt, global_gmm = fit_gmmx(X_train, n_components=n_components, max_iter=max_iter, covariance_type=covariance_type,
                        init_params='best-of-10-pca')
                    ll_global = compute_log_likelihood(global_gmm, X_test)
                    pprint(f"{dt:.2f}s {ll_global:.4f} gmmx-bestof10pcainit {covariance_type} K={n_components} {n_init} {max_iter}")
                    del dt, global_gmm, ll_global

                    dt, global_gmm = fit_gmmx(X_train, n_components=n_components, max_iter=max_iter, covariance_type=covariance_type,
                        init_params='best-of-10-diag')
                    ll_global = compute_log_likelihood(global_gmm, X_test)
                    pprint(f"{dt:.2f}s {ll_global:.4f} gmmx-bestof10diaginit {covariance_type} K={n_components} {n_init} {max_iter}")
                    del dt, global_gmm, ll_global

        for max_iter in 1, 2, 5, 10, 100:
            for n_init in 1, 10:
                for refine_weights, min_weight in (False, 0), (True, 0), (None, 0): #, (True, 2. / len(X_train)):
                    dt, gmmsk = fit_lightgmm(
                        X_train, n_components=n_components, refine_weights=refine_weights,
                        init_kwargs=dict(n_init=n_init, max_iter=max_iter, init='random'), 
                        **(dict(min_weight=min_weight) if min_weight != 0 else {}))
                    ll_fast = compute_log_likelihood(gmmsk, X_test)
                    pprint(f"{dt:.2f}s {ll_fast:.4f} LightGMM K={n_components} {n_init} {max_iter} {refine_weights} {min_weight}")
                    #if refine_weights and max_iter==2 and n_init==1:
                    #    plot_quality(gmmsk, X_test, f'{filename}_lightgmm{"R" if refine_weights else ""}_{n_components}.pdf')
                del n_init
            continue
            n_init = 1
            refine_weights = True
            for init in 'best-of-K-diag',: #'spaced-diagonal', 'spaced-diagonal-bestof2', 
                dt, gmmsk = fit_lightgmm(
                    X_train, n_components=n_components, refine_weights=refine_weights,
                    init_kwargs=dict(n_init=n_init, max_iter=max_iter, init=init))
                ll_fast = compute_log_likelihood(gmmsk, X_test)
                pprint(f"{dt:.2f}s {ll_fast:.4f} LightGMM K={n_components} {n_init} {max_iter} {refine_weights} {init}")


if __name__ == "__main__":
    main()
