"""
Authors: Dror Dotan <dror.dotan@gmail.com>
         Fosca Al Roumi <fosca.al.roumi@gmail.com>
"""

import numpy as np
from mne.stats import spatio_temporal_cluster_1samp_test



#------------------------------------------------------------
def stats_cluster_based_permutation_test(X):
    """
    Statistical test applied across subjects
    """

    # check input
    assert 2 <= X.ndim <= 3, "X must have 2-3 dimensions"
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X

    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(X, out_type='mask', n_permutations=2 ** 12, n_jobs=-1, verbose=False)

    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval

    return np.squeeze(p_values_).T
