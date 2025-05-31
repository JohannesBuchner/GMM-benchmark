import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split
import corner
import joblib
import numpy as np


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
    axes[i,i-1].legend(title="black:before\ngray:after")
    axes[i,i].legend()
    plt.savefig(outfilename.replace('.pdf', '') + '2d.pdf')
    plt.close()
    print(f"plotted to {outfilename.replace('.pdf', '') + '2d.pdf'}")

# Define the core model
from sklearn.preprocessing import PowerTransformer, StandardScaler

def main():
    if len(sys.argv) < 2:
        print("Usage: python gmm_compare.py data.txt.gz")
        return

    filename = sys.argv[1]
    X = np.loadtxt(filename)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=42)
    print(f"Training data: {X_train.shape} Test data: {X_test.shape}")

    #window_size = 0.75
    #step_size = 0.25
    t = X_train[:,5]
    t_test = X_test[:,5]
    Z_train = X_train.copy()
    Z_test = X_test.copy()
    tmid = sorted(t)[::500]
    print(tmid)
    lo = tmid
    hi = tmid[1:] + [t.max() + 0.01]
    for i, (tlo, thi) in enumerate(zip(lo, hi)):
        mask_used = np.logical_and(t >= tlo, t < thi)
        mask = np.logical_and(t >= tlo, t < thi)
        mask_test = np.logical_and(t_test >= tlo, t_test < thi)
        if not mask.any():
            if mask_test.any():
                Z_test[mask_test,:] = transformer.transform(X_test[mask_test,:]) + mean
            continue
        transformer = StandardScaler()
        transformer.fit(X_train[mask_used,:])
        print(i, mask_used.sum(), np.round(tlo, 2), np.round(thi, 2), np.round(transformer.scale_, 2))
        mean = transformer.mean_[None,:]
        Z_train[mask,:] = transformer.transform(X_train[mask,:]) + mean
        if mask_test.any():
            Z_test[mask_test,:] = transformer.transform(X_test[mask_test,:]) + mean
    
    transformer = StandardScaler()
    transformer.fit(X_train)
    mean = transformer.mean_[None,:]
    plot_quality(Z_train, transformer.transform(X_train) + mean, f'{filename}_deproject_simple_init.pdf')

if __name__ == "__main__":
    main()
