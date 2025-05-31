import time
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split
import corner
import joblib
import jax
import jax.numpy as jnp
from jax import random
from sklearn.cluster import KMeans
import numpy as np
from jaxopt import LBFGS


mem = joblib.Memory('.', verbose=False)


def plot_quality(Y, X, outfilename):
    if os.environ.get('PLOT', '1') == '0':
        return
    print("plotting...")
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

# Define the core model

def project_data(X, origin, direction):
    """Project data X onto the line defined by origin and unit direction vector."""
    direction = direction / jnp.linalg.norm(direction)
    return jnp.dot(X - origin, direction)

def normalize(v):
    return v / jnp.linalg.norm(v)

# Create orthonormal basis for orthogonal subspace
# Use QR factorization on random vectors or use SVD
def build_orthonormal_basis(direction: jnp.ndarray) -> jnp.ndarray:
    # v shape: (D,)
    # returns (D, D-1) orthonormal matrix orthogonal to v
    Q, _ = jnp.linalg.qr(jnp.eye(direction.shape[0]) - jnp.outer(direction, direction))
    return Q[:, :-1]

def decompose_X(X, origin, direction):
    direction_unit = normalize(direction)
    t = project_data(X, origin, direction_unit)
    X_proj = origin + t[:, None] * direction_unit
    X_perp = X - X_proj
    basis = build_orthonormal_basis(direction_unit)
    X_perp_plane = X_perp @ basis  # shape (N, D-1)
    return t, X_perp_plane, basis

def sample_from_model(key, t_mean, t_std, params, N):
    """
    Sample N points from the heteroscedastic GMM model.
    
    params: dict with keys
        - origin: (D,)
        - direction: (D,)
        - means: (K, D-1)
        - log_sigma0: (K, D-1)
        - log_ks: (K, D-1)
        - weights: (K,)
    key: jax.random.PRNGKey
    N: int, number of samples
    
    Returns:
    samples: (N, D)
    """
    torigin, means, log_sigma0, log_ks, rs, weights = params

    K, D_minus_1 = means.shape
    # Normalize direction vector
    basis = build_orthonormal_basis(direction)
    # Sample component indices according to weights
    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    t_samples = random.normal(subkey1, shape=(N,)) * t_std + t_mean
    dt = jnp.clip(t_samples - torigin, 0, None)[:, None]
    component_indices = random.categorical(subkey2, jnp.log(weights), shape=(N,))
    
    # For each sample:
    # 1) Sample t ~ Normal(0, 1) -- you can adjust this or use means on t if modeled
    # 2) Sample orthogonal noise with diagonal covariance from log_sigma0 + t * log_ks
    
    # Get per-sample means in the plane: mu + r * (t - torigin)
    chosen_r = rs[component_indices]  # [N, D-1]
    mean_plane = means[component_indices] + jnp.einsum('ij,i->ij', chosen_r, t_samples)
    assert log_sigma0.shape == (D_minus_1,), (log_sigma0.shape, (K, D_minus_1))

    # Compute diagonal stddev for orthogonal dimensions
    log_sigma_total = log_sigma0 + dt * log_ks  # [N, D-1]
    assert log_sigma_total.shape == (N, D_minus_1), (log_sigma_total.shape, (N, D_minus_1))
    std_total = jnp.exp(log_sigma_total)

    print("std_total for sampling:", std_total.shape, std_total.min(axis=0), std_total.max(axis=0))
    # Sample orthogonal noise
    ortho_noise = random.normal(subkey3, shape=(N, D_minus_1)) * std_total
    print("basis", basis)

    ortho_part = (mean_plane + ortho_noise) @ basis.T  # [N, D]
    samples = t_samples[:, None] * direction + ortho_part  # [N, D]    
    return samples

def compute_log_likelihood(X, torigin, mus_plane, log_sigma0, log_ks, rs, weights):
    """
    X: [N, D] data
    origin: [D] the origin point of the projection line
    direction: [D] unit vector for the projection line
    mus_plane: [K, D-1] means in plane orthogonal to direction
    log_sigmas_plane: [K, D-1] log stds for each component
    weights: [K] mixture weights
    """
    N, D = X.shape
    K, D_minus_1 = mus_plane.shape

    # Decompose X into t and plane coords
    t, X_perp_plane, _ = decompose_X(X, 0, direction)  # [N], [N, D-1], _
    dt = jnp.clip(t - torigin, 0, None)[:, None]  # [N, 1]

    def component_log_prob(k):
        # Compute shifted means
        mean_shift = mus_plane[k] + rs[k] * dt  # [N, D-1]
        delta = X_perp_plane - mean_shift  # [N, D-1]

        # Compute log sigma per sample
        log_sigma_k = log_sigma0[k] + dt * log_ks[k]  # [N, D-1]
        log_det = jnp.sum(2 * log_sigma_k, axis=1)  # [N]
        mahal = jnp.sum((delta / jnp.exp(log_sigma_k)) ** 2, axis=1)  # [N]
        return -0.5 * (mahal + log_det) + jnp.log(weights[k])  # [N]

    # compute probability for each component
    log_probs = jax.vmap(component_log_prob)(jnp.arange(K))  # [K, N]
    total_log_prob = jax.scipy.special.logsumexp(log_probs, axis=0)
    return jnp.mean(total_log_prob)

# Initialization function
def initialize_model(X, K):
    N, D = X.shape
    # initial guess: take data center and linear increase
    torigin = 0.5 * jnp.ones(1)

    # Project data
    t = project_data(X, 0, direction)
    direction_unit = direction / jnp.linalg.norm(direction)
    X_proj = jnp.outer(t, direction_unit)
    X_perp = X - X_proj

    # Build orthonormal basis for plane
    Q, _ = jnp.linalg.qr(jnp.eye(D) - jnp.outer(direction_unit, direction_unit))
    X_perp_plane = jnp.dot(X_perp, Q)[:, :-1]  # remove last dim for stability

    # use KMeans in orthogonal plane to guess initial cluster centers
    kmeans = KMeans(n_clusters=K).fit(np.array(X_perp_plane))
    labels = kmeans.labels_
    mus_plane = jnp.array(kmeans.cluster_centers_)  # [K, D-1]

    # Shared initial log std, which is the entire variance.
    # log_sigmas_plane = jnp.log(jnp.std(X_perp_plane, axis=0) * 0 + 0.1)
    residuals = []
    for k in range(K):
        cluster_points = X_perp_plane[labels == k]
        if not cluster_points.shape[0] > 1:
            continue
        residuals.append(cluster_points - mus_plane[k])
    log_sigmas_plane = jnp.log(jnp.std(jnp.concatenate(residuals), axis=0) + 1e-5)
    print("logsigmas:", log_sigmas_plane.shape, jnp.exp(log_sigmas_plane))

    # linear increase of the variance with t
    log_ks = 0.2 * jnp.ones(D - 1)
    rs = jnp.zeros((K, D - 1))
    weights = jnp.ones(K) / K
    
    print('number of model parameters:', torigin.size + mus_plane.size + log_sigmas_plane.size + log_ks.size + rs.size + weights.size)

    return torigin, mus_plane, log_sigmas_plane, log_ks, rs, weights

# Wrapper objective for optimization
def loss_fn(params, X):
    torigin, mus_plane, log_sigmas_plane, log_ks, rs, weights = params
    return -compute_log_likelihood(X, torigin, mus_plane, log_sigmas_plane, log_ks, rs, weights)


# Example usage:
# X is your (N, D) JAX array of data
# origin, direction, mus_plane, log_sigmas_plane, weights = initialize_model(X, K=5)
# params = (origin, direction, mus_plane, log_sigmas_plane, weights)
# loss = loss_fn(params, X)

value_and_grad_fn = jax.value_and_grad(loss_fn)

@mem.cache
def fit_model(X_train, params_init, **kwargs):
    t0 = time.time()
    solver = LBFGS(fun=loss_fn, **kwargs)
    params_opt, state = solver.run(params_init, X_train)
    return time.time() - t0, params_opt, state.value

direction = None

def main():
    if len(sys.argv) < 2:
        print("Usage: python gmm_compare.py data.txt.gz")
        return

    filename = sys.argv[1]
    X = np.loadtxt(filename)
    global direction
    direction = jnp.array(np.where(np.arange(X.shape[1]) == 5, 1.0, 0.0))
    print(direction)
    
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=42)
    print(f"Training data: {X_train.shape} Test data: {X_test.shape}")
    del X
    X_train -= X_train.mean(axis=0, keepdims=True)
    print(f"Loaded data with shape: {X_train.shape}")

    
    t = X_train[:,5]
    params_init = initialize_model(X_train, 2)
    key = jax.random.PRNGKey(42)
    samples = sample_from_model(key, t.mean(), t.std(), params_init, 100000)
    plot_quality(np.array(samples), X_train, f'{filename}_deproject_init.pdf')
    del samples
    dt, params_opt, loss = fit_model(X_train, params_init, maxiter=1000, tol=1e-3)
    print(f"{dt:2f}s")
    print("Optimized loss:", loss)
    print("Optimized params:", params_opt)
    samples = sample_from_model(key, t.mean(), t.std(), params_opt, 100000)
    mask = np.isfinite(samples).all(axis=0)
    plot_quality(np.array(samples[mask,:]), X_train, f'{filename}_deproject_fit.pdf')


if __name__ == "__main__":
    main()
