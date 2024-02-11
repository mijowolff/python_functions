

import numpy as np
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