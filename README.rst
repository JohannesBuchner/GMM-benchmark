GMM benchmark
-------------

This benchmark tests GMM libraries for a particular requirement:

* dimensionality: 14-15d
* 50000 training data samples (and another version trained with only 5000)

There are 49 data sets (COSMOS_z[0-9]*pred_mags.txt.gz).

Exploratory data analysis
-------------------------

The data are challenging because the distribution is 

* high-dimensional
* multi-modal
* funnel-shaped

See an illustration of the data distribution in
COSMOS_z0.3-0.5_pred_mags.txt.gz_gmm_globalfull_802d.png:

.. image:: https://raw.githubusercontent.com/JohannesBuchner/GMM-benchmark/refs/heads/main/COSMOS_z0.3-0.5_pred_mags.txt.gz_gmm_globalfull_802d.png
    :width: 700

Objective
---------

Maximize both:

1. training time (with a goal of <100ms per data set)
2. model quality, as measured by test set average loglikelihood (score).

There is also a constraint to only use models where conditional likelihoods can be evaluated. This boils down to Gaussian mixture models at the moment.

I use 'full' covariance matrices, because 'diag' and 'tied' gave very poor loglikelihood results, even with many more components.

Libraries tested
----------------

Library -> model variants/configurations are:

* Scikit-learn 
  * GaussianMixture (EM algorithm), initialised with kmeans++
  * BayesianGaussianMixture (Variational) (but discarded, because not better than GaussianMixture nor faster)
* `gmmx <https://github.com/adonath/gmmx>`_, GMM+EM 
  * initialised with kmeans++
  * "fastinit": random sample initialisation, 1 K-Means iteration, 1 EM iteration. The fork is at: https://github.com/JohannesBuchner/gmmx/
* LightGMM (in https://github.com/JohannesBuchner/askcarl/ under askcarl.lightgmm): This does not use the EM algorithm at all, but random sample initialisation and then KMeans steps (n_iter max_iter) give the cluster centers. Regularized covariance matrices are computed from cluster members. The component weights are assigned either
  * 'Equal' (this works poorly)
  * 'Kmeans' (proportional to number of cluster members)
  * 'Refine' (a single E step, maximizing the training set likelihood)

In all models, the number of components K are varied: 5, 10, 15, 20, 30, 40, 80, 120. max_iter and n_init are also varied.

I also tested:

* Mixture-Models library "https://github.com/kasakh/Mixture-Models/"
  * PGMM, gave very poor results and was very slow
  * MFA, gave very poor results and was very slow

Results
=======

See *.data files, which give duration, score (average log-likelihood on test set) and model name. The results are listed as: "{modelname} K={n_components} {n_init} {max_iter}"

Pareto frontier
---------------

50000 samples:

.. image:: https://raw.githubusercontent.com/JohannesBuchner/GMM-benchmark/refs/heads/main/pareto_analysis_lightgmm50000.png
    :width: 700
    :target: https://raw.githubusercontent.com/JohannesBuchner/GMM-benchmark/refs/heads/main/pareto_analysis_lightgmm50000.pdf

5000 samples:

.. image:: https://raw.githubusercontent.com/JohannesBuchner/GMM-benchmark/refs/heads/main/pareto_analysis_lightgmm5000.png
    :width: 700
    :target: https://raw.githubusercontent.com/JohannesBuchner/GMM-benchmark/refs/heads/main/pareto_analysis_lightgmm5000.pdf


Summary
-------

1. At highest speeds and large data set, LightGMM surpasses gmmx in quality, even with fastinit.
2. gmmx does equally well or better with fastinit for the smaller data set.
3. For the highest reconstruction quality, EM iterations are needed.

Overall good performance at the sub-second level:

* LightGMM with 80 components for 50000 samples
* LightGMM with 40 components for 5000 samples (tied with gmmx-fastinit full K=10-30 n_iter=1 max_iter=1)

