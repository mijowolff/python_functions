

import numpy as np

from scipy.ndimage import label, find_objects
#%% permutation t test as implemented in mne, but without correction


def _surrogate_stat(X, X2, perms, dof_scaling):

    from math import sqrt


    n_samples = len(X)
    mus = np.dot(perms, X) / float(n_samples)
    stds = np.sqrt(X2[None, :] - mus * mus) * dof_scaling  # std with splitting
    surrogate_abs = np.abs(mus) / (stds / sqrt(n_samples))  # surrogates
    return surrogate_abs
 

def permutation_t_test_no_correction(X, n_permutations=10000, tail=0, n_jobs=1,seed=None): 

    from math import sqrt
    from sklearn.utils import check_random_state
    from python_utils import logger
    from statsmodels.tools.parallel import parallel_func
    from mne.stats.cluster_level import _get_1samp_orders    

    n_samples, n_tests = X.shape
    import numpy as np
    X2 = np.mean(X ** 2, axis=0)  # precompute moments
    mu0 = np.mean(X, axis=0)
    dof_scaling = sqrt(n_samples / (n_samples - 1.0))
    std0 = np.sqrt(X2 - mu0 ** 2) * dof_scaling  # get std with var splitting
    T_obs = np.mean(X, axis=0) / (std0 / sqrt(n_samples))
    rng = check_random_state(seed)
    orders, _, extra = _get_1samp_orders(n_samples, n_permutations, tail, rng)
    perms = 2 * np.array(orders) - 1  # from 0, 1 -> 1, -1
    # logger.info('Permuting %d times%s...' % (len(orders), extra))
    parallel, my_surrogate_stat, n_jobs = parallel_func(_surrogate_stat, n_jobs)
    surrogate_abs = np.concatenate(parallel(my_surrogate_stat(X, X2, p, dof_scaling)
                                      for p in np.array_split(perms, n_jobs)))
    surrogate_abs = np.concatenate((surrogate_abs, np.abs(T_obs[np.newaxis, :])))
    H0 = np.sort(surrogate_abs, axis=0)
    # compute UNCORRECTED p-values
    if tail == 0:
        p_values = (H0 >= np.abs(T_obs)).mean(axis=0)
    elif tail == 1:
        p_values = (H0 >= T_obs).mean(axis=0)
    elif tail == -1:
        p_values = (-H0 <= T_obs).mean(axis=0)
    return T_obs, p_values, H0

#%%
# convert 1d matlab cell-array to list

def matcell1d_to_list(matcell1d):
    matcell1d=np.squeeze(matcell1d)
    
    new_list=[]   
    for ind in range(len(matcell1d)):
        temp=matcell1d[ind]
        ind2=temp.tolist()
        new_list.append(ind2[0])
        
    return new_list

#%%

def circ_mean(alpha, axis=None, w=None):
    if w is None:
        w = np.ones(alpha.shape)

    # weight sum of cos & sin angles
    t = w * np.exp(1j * alpha)

    r = np.sum(t, axis=axis)
    
    mu = np.angle(r)

    return mu
#%%

def circ_dist(x,y,all_pairs=False):
    
    # circular distance between angles in radians
    
    x=np.asarray(x)
    y=np.asarray(y)
    
    x=np.squeeze(x)
    y=np.squeeze(y)
    
    if all_pairs:
        x_new=np.tile(np.exp(1j*x),(len(y),1))
        y_new=np.transpose(np.tile(np.exp(1j*y),(len(x),1)))
        circ_dists= np.angle(x_new/y_new)
    else:
        circ_dists= np.angle(np.exp(1j*x)/np.exp(1j*y))
        
    return circ_dists

#%%
def cluster_test(datobs, datrnd, tail=None, alpha=0.05, clusteralpha=0.05, clusterstat='sum'):
    """
    CLUSTER_TEST performs a cluster-corrected test that datobs is higher/lower
    than the distribution as expected under the null hypothesis. The 'null'
    distribution should be pre-computed (manually or using CLUSTER_TEST_HELPER)
    and entered as an argument into this function.

    datobs - observed data MxNx...xZ
    datrnd - null distribution, MxNx...xZxPerm
    tail - whether to test datobs < null (tail==-1), datobs > null (tail==1)
    or datobs <> null (tail==0, default).
    alpha - critical level (default 0.05)
    clusteralpha - nonparametric threshold for cluster candidates (default 0.05)
    clusterstat - how to combine statistics in cluster candidates (can be
    'sum' (default) or 'size')

    Returns:
    h - MxNx...xZ logical matrix indicating where significant clusters were
    found (though note that formally speaking the test concerns the data as a
    whole, so the interpretation of the location of clusters within h should
    be done with caution).
    p - MxNx...xZ matrix of p-values associated with clusters.
    clusterinfo - struct with extra cluster info, e.g. indices

    Originally written in Matlab by Eelke Spaak, 2015
    """

    # Defaults handling
    if tail is None:
        tail = 0

    if clusterstat not in ['sum', 'size']:
        raise ValueError("Unsupported clusterstat")

    # Some bookkeeping
    rndsiz = datrnd.shape
    rnddim = len(rndsiz)
    numrnd = rndsiz[rnddim - 1]

    if not (np.all(datobs.shape == rndsiz[:-1]) or (np.ndim(datobs) == 1 and len(datobs) == rndsiz[0])):
        raise ValueError("datobs and datrnd are not of compatible dimensionality")

    # Handle different options for computing cluster characteristics
    cluster_stat_sum = clusterstat == 'sum'
    cluster_stat_size = clusterstat == 'size'

    # Actual cluster test
    clusteralpha = clusteralpha / 2 if tail == 0 else clusteralpha

    cluster_threshold_neg = np.quantile(datrnd, clusteralpha, axis=rnddim - 1)
    cluster_threshold_pos = np.quantile(datrnd, 1 - clusteralpha, axis=rnddim - 1)

    clus_observed_pos, pos_inds = find_and_characterize_clusters(datobs, datobs >= cluster_threshold_pos)
    clus_observed_neg, neg_inds = find_and_characterize_clusters(datobs, datobs <= cluster_threshold_neg)

    null_pos = np.empty(numrnd)
    null_neg = np.empty(numrnd)

    for k in range(numrnd):
        clus_rnd_pos, _ = find_and_characterize_clusters(datrnd[..., k], datrnd[..., k] >= cluster_threshold_pos)
        clus_rnd_neg, _ = find_and_characterize_clusters(datrnd[..., k], datrnd[..., k] <= cluster_threshold_neg)
        
        null_pos[k] = np.max(clus_rnd_pos) if clus_rnd_pos.size else np.nan
        null_neg[k] = np.min(clus_rnd_neg) if clus_rnd_neg.size else np.nan

    null_pos = null_pos[~np.isnan(null_pos)]
    null_neg = null_neg[~np.isnan(null_neg)]

    null_pos = -np.sort(-null_pos)
    null_neg = np.sort(null_neg)

    clus_p_pos = np.ones_like(clus_observed_pos)
    clus_p_neg = np.ones_like(clus_observed_neg)

    for i, obs_pos in enumerate(clus_observed_pos):
        clus_p_pos[i] = (np.sum(null_pos > obs_pos) + 1) / (numrnd + 1)

    for i, obs_neg in enumerate(clus_observed_neg):
        clus_p_neg[i] = (np.sum(null_neg < obs_neg) + 1) / (numrnd + 1)

    # if np.ndim(datobs) == 1:
    #     p = np.ones_like(datobs)
    # else:
    #     p = np.ones(datobs.shape[:-1])

    p = np.ones_like(datobs)

    clusterinfo = {'pos_clusters': [], 'neg_clusters': []}

    if tail >= 0:
        for i, pos_ind in enumerate(pos_inds):
            p[pos_ind] = clus_p_pos[i]

            clusterinfo['pos_clusters'].append({
                'clusterstat': clus_observed_pos[i],
                'p': clus_p_pos[i],
                'inds': np.zeros_like(datobs).astype(bool)
            })
            clusterinfo['pos_clusters'][i]['inds'][pos_ind] = True

            if tail == 0:
                clusterinfo['pos_clusters'][i]['p'] *= 2

    if tail <= 0:
        for i, neg_ind in enumerate(neg_inds):
            p[neg_ind] = clus_p_neg[i]

            clusterinfo['neg_clusters'].append({
                'clusterstat': clus_observed_neg[i],
                'p': clus_p_neg[i],
                'inds': np.zeros_like(datobs).astype(bool)
            })
            clusterinfo['neg_clusters'][i]['inds'][neg_ind] = True

            if tail == 0:
                clusterinfo['neg_clusters'][i]['p'] *= 2

    if tail == 0:
        p = np.minimum(1, p * 2)

    h = p < alpha

    return h, p, clusterinfo

def find_and_characterize_clusters(dat, clus_cand):
    labeled_array, num_features = label(clus_cand)
    objects = find_objects(labeled_array)

    clus_stats = []
    inds = []

    for obj in objects:
        mask = labeled_array[obj] == (np.arange(1, num_features + 1)[..., np.newaxis, np.newaxis])
        clus_stats.append(np.sum(dat[obj] * mask) if np.sum(mask) > 1 else 0)
        inds.append(obj)
    
    # convert to numpy array
    clus_stats = np.array(clus_stats)

    return clus_stats, inds

#%%
import numpy as np

def cluster_test_helper(dat, nperm, diffstat='diff',verbose=True):
    """
    CLUSTER_TEST_HELPER is a helper function for doing cluster-corrected
    permutation tests in arbitrary dimensions. The randomizations are
    generated under the assumption that input dat was computed using a paired
    statistic T for which T(a,b) = -T(b,a) holds, where a and b are the data
    under the two paired conditions. (E.g. raw difference would work.)

    dat - an NxMx...xZxObs data matrix. The trailing dimension must correspond
    to the unit of observation (e.g., subjects or trials).

    nperm - the number of permutations to generate

    diffstat - how to compute the 'difference statistic'. Can be 'diff'
    (default) or 't' (compute one-sample t-score).

    Returns:
    datobs - NxMx...xZ statistic for observed data, averaged across observations
    datrnd - NxMx...xZxPerm statistic under the null hypothesis

    Originally written in Matlab by Eelke Spaak, 2015
    """

    print("generating randomization distribution, assuming dat was generated")
    print("using a paired test statistic T for which T(a,b) = -T(b,a) holds...")

    usetscore = diffstat == 't'

    # get data characteristics
    siz = dat.shape
    sampdim = len(siz)
    nSamp = siz[sampdim - 1]
    sqrtNSamp = np.sqrt(nSamp)

    # the observed statistic: mean across the observed samples
    mu = np.mean(dat, axis=sampdim - 1)
    if usetscore:
        sd = np.std(dat, axis=sampdim - 1)
        datobs = mu / (sd / sqrtNSamp)
    else:
        datobs = mu

    # initialize space for randomization distribution
    siz = datobs.shape
    siz = siz + (nperm,)
    datrnd = np.zeros(siz)
    rnddim = len(siz)

    for k in range(nperm):
        if verbose:
            if k % round(nperm / 10) == 0:
                print("generating permutation", k, "of", nperm, "...")

        # copy the data
        tmp = np.copy(dat)

        # randomly flip the sign of ~half the observations
        # this is probably a slightly overly conservative way of estimating the
        # null distribution
        flipinds = np.random.rand(nSamp) < 0.5

        tmp[..., flipinds] *= -1

        # store the mean across the surrogate data in the null distribution
        if usetscore:
            mu = np.mean(tmp, axis=rnddim - 1)
            sd = np.std(tmp, axis=rnddim - 1)
            datrnd[..., k] = mu / (sd / sqrtNSamp)
        else:
            datrnd[..., k] = np.mean(tmp, axis=rnddim - 1)
    
    if verbose:
        print("done.")

    return datobs, datrnd