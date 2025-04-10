
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.spatial import distance
import numpy as np
import random
from numpy.linalg import inv
import warnings


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

def covdiag(x):
    
    '''
    x (t*n): t iid observations on n random variables
    sigma (n*n): invertible covariance matrix estimator
    
    Shrinks towards diagonal matrix
    as described in Ledoit and Wolf, 2004
    '''
    
    t,n=np.shape(x)
    
    # de-mean
    x=x-np.mean(x,axis=0)
    
    #get sample covariance matrix
    sample=np.cov(x,rowvar=False,bias=True)
    
    #compute prior
    prior=np.zeros((n,n))
    np.fill_diagonal(prior,np.diag(sample))
    
    #compute shrinkage parameters
    d=1/n*np.linalg.norm(sample-prior,ord='fro')**2
    y=x**2
    r2=1/n/t**2*np.sum(np.dot(y.T,y))-1/n/t*np.sum(sample**2)
    
    #compute the estimator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shrinkage=max(0,min(1,r2/d))
    sigma=shrinkage*prior+(1-shrinkage)*sample
    
    return sigma

#%%
def cosfun(theta,mu,basis_smooth,amplitude='default',offset='default'):

    if amplitude=='default':
        amplitude=.5
    if offset=='default':
        offset=.5
    return (offset+amplitude*np.cos(theta-mu))**basis_smooth
#%%
def basis_set_fun(theta_bins,u_theta,basis_smooth='default'):
        
    if basis_smooth=='default':
        basis_smooth=theta_bins.shape[0]-1
        
    smooth_bins=np.zeros(theta_bins.shape)
    
    for ci in range(theta_bins.shape[0]):
        temp_kernel=cosfun(u_theta,u_theta[ci],basis_smooth)
        temp_kernel=np.expand_dims(temp_kernel,axis=[1,2])
        temp_kernel=np.tile(temp_kernel,(1,theta_bins.shape[1],theta_bins.shape[2]))
        smooth_bins[ci,:,:]=np.sum(theta_bins*temp_kernel,axis=0)/sum(temp_kernel)                        
    
    return smooth_bins


#%%  distance-based orientation decoding using cross-validation
def dist_theta_kfold(data,theta,n_folds=8,n_reps=10,data_trn=None,basis_set=True,angspace='default',ang_steps=4,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True):
    
    if verbose:
        from progress.bar import ChargingBar

    if data_trn is None:
        data_trn=data
        
    if type(angspace)==str:
        if angspace=='default':
            angspace=np.arange(-np.pi,np.pi,np.pi/8) # default is 16 bins
        
    if np.array_equal(angspace,np.unique(theta)):
        ang_steps=1        
                
    bin_width=np.diff(angspace)[0]
    
    x_dummy=np.zeros(len(theta)) # needed for sklearn splitting function
    
    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
            
    ntrls, nchans, ntps=np.shape(X_ts)  

    m_temp=np.zeros((len(angspace),nchans,ntps))
    m=m_temp
      
    if verbose:
        bar = ChargingBar('Processing', max=ntps*ang_steps*n_reps*n_folds)
    
    distances=np.empty((ang_steps,len(angspace),ntrls,ntps))
    
    distances[:]=np.NaN

    angspaces=np.zeros((ang_steps,len(angspace)))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
    
        angspace_temp=angspace+ans*bin_width/ang_steps
        angspaces[ans,:]=angspace_temp

    angspace_full=np.reshape(angspaces,(angspaces.shape[0]*angspaces.shape[1]),order='F')

    theta_dists=circ_dist(angspace_full,theta,all_pairs=True)
    theta_dists=theta_dists.transpose()  

    theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
    theta_dists2=np.tile(theta_dists_temp,(1,1,ntps))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
    
        angspace_temp=angspace+ans*bin_width/ang_steps
        
        # convert orientations into bins
        temp=np.argmin(abs(circ_dist(angspace_temp,theta,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta),1))               
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        
        y_subst=temp
        y=bin_orient_rads
                
        rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps) # get splitting object
        
        split_counter=0
        
        distances_temp=np.empty([len(angspace_temp),ntrls,n_reps,ntps])
        distances_temp[:]=np.NaN
               
        for train_index, test_index in rskf.split(X=x_dummy,y=y_subst): # loop over all train/test folds, and repepitions
            
            X_train, X_test = X_tr[train_index,:,:], X_ts[test_index,:,:]
            y_train, y_test = y[train_index], y[test_index]
            y_subst_train, y_subst_test = y_subst[train_index], y_subst[test_index]
                        
            irep=int(np.floor(split_counter/n_folds))
            split_counter=split_counter+1
          
            train_dat_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
            train_dat_cov[:]=np.NaN
            
            if balanced_train_bins: # average over same orientaions of training set, but make sure these averages are based on balanced trials
                count_min=min(np.bincount(y_subst_train))
                for c in range(len(angspace_temp)):
                    temp_dat=X_train[y_train==angspace_temp[c],:,:]
                    ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                    m_temp[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                    if balanced_cov: # if desired, the data used for the covariance can also be balanced
                        if residual_cov: # take the residual, note that this should only be done if the cov data is balanced!
                            train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                        else:
                            train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
            else:
                for c in range(len(angspace_temp)):
                    m_temp[c,:,:]=np.mean(X_train[y_train==angspace_temp[c],:,:],axis=0)
                    
            if basis_set: # smooth the averaged train data with basis set
                m=basis_set_fun(m_temp,angspace_temp,basis_smooth='default')
            else:
                m=m_temp
            
            if not balanced_cov:
                train_dat_cov=X_train # use all train trials if cov is not balanced                       
                    
            for tp in range(ntps):
                m_train_tp=m[:,:,tp]
                X_test_tp=X_test[:,:,tp]
                
                if dist_metric=='mahalanobis':
                    dat_cov_tp=train_dat_cov[:,:,tp]
                    if new_version: # with a lot of dimensions, first performing pca and then using euclidian distance is faster (when using cdist)
                        cov=covdiag(dat_cov_tp) # use covariance of the training data for pca
                        train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                        X_test_tp_centered = X_test_tp - train_dat_cov_avg
                        m_train_tp_centered = m_train_tp -train_dat_cov_avg
                        evals,evecs = np.linalg.eigh(cov)
                        idx = evals.argsort()[::-1]
                        evals = evals[idx]
                        evecs = evecs[:,idx]
                        evals_sqrt = np.sqrt(evals)

                        # compute euclidan distance in whitented pca space (which is identical to mahalanobis distance)
                        distances_temp[:,test_index,irep,tp] = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_tp_centered,evecs)/evals_sqrt, 'euclidean')
                    else:
                        cov=inv(covdiag(dat_cov_tp)) 
                        distances_temp[:,test_index,irep,tp]=distance.cdist(m_train_tp,X_test_tp,'mahalanobis', VI=cov) # compute distances between all test trials, and average train trials
                else:                    
                    distances_temp[:,test_index,irep,tp]=distance.cdist(m_train_tp,X_test_tp,'euclidean')
                   
                if verbose:    
                    bar.next()

        distances[ans,:,:,:]=np.mean(distances_temp,axis=2,keepdims=False)
    
    distances=distances-np.mean(distances,axis=1,keepdims=True)
    distances_flat=np.reshape(distances,(distances.shape[0]*distances.shape[1],distances.shape[2],distances.shape[3]),order='F')
    distances_flat=distances_flat-np.mean(distances_flat,axis=0,keepdims=True)
    dec_cos=np.squeeze(-np.mean(np.cos(theta_dists2)*distances_flat,axis=0))

    # order the distances, such that same angle distances are in the middle
    # first, assign each theta to a bin from angspace_full
    temp=np.argmin(abs(circ_dist(angspace_full,theta,all_pairs=True)),axis=1)
    ang_bin_temp=np.tile(angspace_full,(len(theta),1))               
    theta_bins=ang_bin_temp[:,temp][0,:]

    # then, sort the distances based on the distances between the theta_bins
    theta_bin_dists=circ_dist(angspace_full,theta_bins,all_pairs=True)
    theta_bin_dists=theta_bin_dists.transpose()
    theta_bin_dists_abs=np.abs(theta_bin_dists)
    # get index of the minimum distance
    theta_bin_dists_min_ind=np.argmin(theta_bin_dists_abs,axis=0)

    distances_ordered=np.zeros((distances_flat.shape))

    shift_to=np.where(angspace_full==0)[0][0]
    for trl in range(len(theta)):
        distances_ordered[:,trl,:] = np.roll(distances_flat[:,trl,:], int(shift_to - theta_bin_dists_min_ind[trl]), axis=0)
    
    if verbose:
        bar.finish()
    
    return dec_cos,distances,distances_ordered,angspaces,angspace_full
#%%  distance-based orientation decoding using cross-validation
# special version where the stratified kfold uses not only the orientation bins, but also a second, unrelated variable
def dist_theta_kfold_special(data,theta,n_folds=8,n_reps=10,data_trn=None,basis_set=True,angspace='default',ang_steps=4,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True,mc_trls=True,mc_bins=True,second_var=0,strat_full=False):
    
    if verbose:
        from progress.bar import ChargingBar

    if data_trn is None:
        data_trn=data
        
    if type(angspace)==str:
        if angspace=='default':
            angspace=np.arange(-np.pi,np.pi,np.pi/8) # default is 16 bins
        
    if np.array_equal(angspace,np.unique(theta)):
        ang_steps=1        
                
    bin_width=np.diff(angspace)[0]
    
    x_dummy=np.zeros(len(theta)) # needed for sklearn splitting function
    
    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
            
    ntrls, nchans, ntps=np.shape(X_ts)  

    m_temp=np.zeros((len(angspace),nchans,ntps))
    m=m_temp
      
    if verbose:
        bar = ChargingBar('Processing', max=ntps*ang_steps*n_reps*n_folds)
    
    distances=np.empty((ang_steps,len(angspace),ntrls,ntps))
    
    distances[:]=np.NaN

    angspaces=np.zeros((ang_steps,len(angspace)))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
        angspace_temp=angspace+ans*bin_width/ang_steps
        angspaces[ans,:]=angspace_temp

    angspace_full=np.reshape(angspaces,(angspaces.shape[0]*angspaces.shape[1]),order='F')

    theta_dists=circ_dist(angspace_full,theta,all_pairs=True)
    theta_dists=theta_dists.transpose()  

    theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
    theta_dists2=np.tile(theta_dists_temp,(1,1,ntps))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
    
        angspace_temp=angspace+ans*bin_width/ang_steps
        
        # convert orientations into bins
        temp=np.argmin(abs(circ_dist(angspace_temp,theta,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta),1))               
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        
        y_subst=temp
        y_subst2=temp+second_var
        # convert to unique integers starting from 0, and going in steps of 1
        y_subst2=np.unique(y_subst2,return_inverse=True)[1]

        if strat_full:
            y_subst=y_subst2

        y=bin_orient_rads
                
        rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps) # get splitting object
        
        split_counter=0
        
        distances_temp=np.empty([len(angspace_temp),ntrls,n_reps,ntps])
        distances_temp[:]=np.NaN
               
        for train_index, test_index in rskf.split(X=x_dummy,y=y_subst2): # loop over all train/test folds, and repepitions
            
            X_train, X_test = X_tr[train_index,:,:], X_ts[test_index,:,:]
            y_train, y_test = y[train_index], y[test_index]
            y_subst_train, y_subst_test = y_subst[train_index], y_subst[test_index]
                        
            irep=int(np.floor(split_counter/n_folds))
            split_counter=split_counter+1
          
            train_dat_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
            train_dat_cov[:]=np.NaN

            train_dat_res_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
            train_dat_res_cov[:]=np.NaN
            
            if balanced_train_bins: # average over same orientaions of training set, but make sure these averages are based on balanced trials
                count_min=min(np.bincount(y_subst_train))
                for c in range(len(angspace_temp)):
                    temp_dat=X_train[y_train==angspace_temp[c],:,:]
                    ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                    m_temp[c,:,:]=np.nanmean(temp_dat[ind,:,:],axis=0)
                    if balanced_cov: # if desired, the data used for the covariance can also be balanced
                        if residual_cov: # take the residual, note that this should only be done if the cov data is balanced!
                            train_dat_res_cov = np.append(train_dat_res_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                        train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
            else:
                for c in range(len(angspace_temp)):
                    m_temp[c,:,:]=np.nanmean(X_train[y_train==angspace_temp[c],:,:],axis=0)
                    
            if basis_set: # smooth the averaged train data with basis set
                m=basis_set_fun(m_temp,angspace_temp,basis_smooth='default')
            else:
                m=m_temp
            
            if not balanced_cov:
                train_dat_cov=X_train # use all train trials if cov is not balanced  

            if np.isnan(train_dat_res_cov).all():
                train_dat_res_cov=train_dat_cov                     
                    
            for tp in range(ntps):
                m_train_tp=m[:,:,tp]
                X_test_tp=X_test[:,:,tp]
                
                if dist_metric=='mahalanobis':
                    dat_cov_tp=train_dat_cov[:,:,tp]
                    dat_cov_res_tp=train_dat_res_cov[:,:,tp]
                    if new_version: # with a lot of dimensions, first performing pca and then using euclidian distance is faster (when using cdist)
                        cov=covdiag(dat_cov_res_tp) # use covariance of the training data for pca
                        train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                        X_test_tp_centered = X_test_tp - train_dat_cov_avg
                        m_train_tp_centered = m_train_tp -train_dat_cov_avg
                        evals,evecs = np.linalg.eigh(cov)
                        idx = evals.argsort()[::-1]
                        evals = evals[idx]
                        evecs = evecs[:,idx]
                        evals=evals.clip(1e-10) # avoid division by zero
                        evals_sqrt = np.sqrt(evals)
                        # compute euclidan distance in whitented pca space (which is identical to mahalanobis distance)
                        distances_temp[:,test_index,irep,tp] = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_tp_centered,evecs)/evals_sqrt, 'euclidean')
                    else:
                        cov=inv(covdiag(dat_cov_tp)) 
                        distances_temp[:,test_index,irep,tp]=distance.cdist(m_train_tp,X_test_tp,'mahalanobis', VI=cov) # compute distances between all test trials, and average train trials
                else:                    
                    distances_temp[:,test_index,irep,tp]=distance.cdist(m_train_tp,X_test_tp,'euclidean')
                   
                if verbose:    
                    bar.next()

        distances[ans,:,:,:]=np.nanmean(distances_temp,axis=2,keepdims=False)
    
    if mc_trls:
        distances=distances-np.nanmean(distances,axis=1,keepdims=True) # mean-center across trials
    distances_flat=np.reshape(distances,(distances.shape[0]*distances.shape[1],distances.shape[2],distances.shape[3]),order='F')
    if mc_bins:
        distances_flat=distances_flat-np.nanmean(distances_flat,axis=0,keepdims=True) # mean-center across bins
    dec_cos=np.squeeze(-np.nanmean(np.cos(theta_dists2)*distances_flat,axis=0))

    # order the distances, such that same angle distances are in the middle
    # first, assign each theta to a bin from angspace_full
    temp=np.argmin(abs(circ_dist(angspace_full,theta,all_pairs=True)),axis=1)
    ang_bin_temp=np.tile(angspace_full,(len(theta),1))               
    theta_bins=ang_bin_temp[:,temp][0,:]

    # then, sort the distances based on the distances between the theta_bins
    theta_bin_dists=circ_dist(angspace_full,theta_bins,all_pairs=True)
    theta_bin_dists=theta_bin_dists.transpose()
    theta_bin_dists_abs=np.abs(theta_bin_dists)
    # get index of the minimum distance
    theta_bin_dists_min_ind=np.argmin(theta_bin_dists_abs,axis=0)

    distances_ordered=np.zeros((distances_flat.shape))

    shift_to=np.where(np.round(angspace_full,10)==0)[0][0]
    for trl in range(len(theta)):
        distances_ordered[:,trl,:] = np.roll(distances_flat[:,trl,:], int(shift_to - theta_bin_dists_min_ind[trl]), axis=0)
    
    if verbose:
        bar.finish()
    
    return dec_cos,distances,distances_ordered,angspaces,angspace_full
#%%  orientation resconstrution using cross-validation, cross-temporal
def dist_theta_kfold_ct(data,theta,n_folds=8,n_reps=10,data_trn=None,basis_set=True,angspace='default',ang_steps=4,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True):
    
    if verbose:
        from progress.bar import ChargingBar

    if data_trn is None:
        data_trn=data
        
    if type(angspace)==str:
        if angspace=='default':
            angspace=np.arange(-np.pi,np.pi,np.pi/8) # default is 16 bins
        
    if np.array_equal(angspace,np.unique(theta)):
        ang_steps=1        
                
    bin_width=np.diff(angspace)[0]
    
    x_dummy=np.zeros(len(theta)) # needed for sklearn splitting function
    
    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
            
    ntrls, nchans, ntps=np.shape(X_ts)  
    _,_,ntps_trn=np.shape(X_tr)

    m_temp=np.zeros((len(angspace),nchans,ntps))
    m=m_temp
    
    if dist_metric=='euclidean':
        cov_metric=False 
    
    if verbose:
        bar = ChargingBar('Processing', max=ang_steps*n_reps*n_folds*ntps)
    
    dec_cos=np.empty((ang_steps,ntrls,ntps_trn,ntps))
    distances=np.empty((ang_steps,len(angspace),ntrls,ntps_trn,ntps))
    
    dec_cos[:]=np.NaN
    distances[:]=np.NaN

    angspaces=np.zeros((ang_steps,len(angspace)))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
    
        angspace_temp=angspace+ans*bin_width/ang_steps
        angspaces[ans,:]=angspace_temp

    angspace_full=np.reshape(angspaces,(angspaces.shape[0]*angspaces.shape[1]),order='F')

    theta_dists=circ_dist(angspace_full,theta,all_pairs=True)
    theta_dists=theta_dists.transpose()  

    # add two dimensions to theta_dists
    theta_dists_temp=np.expand_dims(theta_dists,axis=-1)

    theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
    theta_dists2=np.tile(theta_dists_temp,(1,1,ntps_trn,ntps))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
    
        angspace_temp=angspace+ans*bin_width/ang_steps
        
        # convert orientations into bins
        temp=np.argmin(abs(circ_dist(angspace_temp,theta,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta),1))               
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        
        y_subst=temp
        y=bin_orient_rads
                
        rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps) # get splitting object
        
        split_counter=0
        
        distances_temp=np.empty([len(angspace_temp),ntrls,n_reps,ntps_trn,ntps])
        distances_temp[:]=np.NaN
        
        theta_dists=circ_dist(angspace_temp,y,all_pairs=True)
        theta_dists=theta_dists.transpose()                
        
        # theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
        # theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
        # theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
        # theta_dists2=np.tile(theta_dists_temp,(1,1,n_reps,ntps,ntps))
        theta_dists=np.tile(np.expand_dims(theta_dists,axis=-1),(1,1,ntps))
        
        angspace_dist=np.unique(np.round(theta_dists,10))
        if -angspace_dist[-1]==angspace_dist[0]:
            angspace_dist=np.delete(angspace_dist,len(angspace_dist)-1)
               
        for train_index, test_index in rskf.split(X=x_dummy,y=y_subst): # loop over all train/test folds, and repepitions
            
            X_train, X_test = X_tr[train_index,:,:], X_ts[test_index,:,:]
            y_train, y_test = y[train_index], y[test_index]
            y_subst_train, y_subst_test = y_subst[train_index], y_subst[test_index]
                        
            irep=int(np.floor(split_counter/n_folds))
            split_counter=split_counter+1
          
            train_dat_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
            train_dat_cov[:]=np.NaN
            
            if balanced_train_bins: # average over same orientaions of training set, but make sure these averages are based on balanced trials
                count_min=min(np.bincount(y_subst_train))
                for c in range(len(angspace_temp)):
                    temp_dat=X_train[y_train==angspace_temp[c],:,:]
                    ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                    m_temp[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                    if balanced_cov: # if desired, the data used for the covariance can also be balanced
                        if residual_cov: # take the residual, note that this should only be done if the cov data is balanced!
                            train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                        else:
                            train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
            else:
                for c in range(len(angspace_temp)):
                    m_temp[c,:,:]=np.mean(X_train[y_train==angspace_temp[c],:,:],axis=0)
                    
            if basis_set: # smooth the averaged train data with basis set
                m=basis_set_fun(m_temp,angspace_temp,basis_smooth='default')
            else:
                m=m_temp
            
            if not balanced_cov:
                train_dat_cov=X_train # use all train trials if cov is not balanced    
                               
            # reshape test data for efficient distance computation
            X_test_rs=np.moveaxis(X_test,-1,1)
            X_test_rs=np.reshape(X_test_rs,(len(test_index)*ntps,X_test.shape[1]),order='C')

            for tp in range(ntps_trn):
                m_train_tp=m[:,:,tp]
                       
                if dist_metric=='mahalanobis':
                    dat_cov_tp=train_dat_cov[:,:,tp]
                    if new_version: # with a lot of dimensions, first performing pca and then using euclidian distance is faster (when using cdist)
                        cov=covdiag(dat_cov_tp) # use covariance of the training data for pca
                        train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                        X_test_rs_centered = X_test_rs - train_dat_cov_avg
                        m_train_tp_centered = m_train_tp -train_dat_cov_avg
                        evals,evecs = np.linalg.eigh(cov)
                        idx = evals.argsort()[::-1]
                        evals = evals[idx]
                        evecs = evecs[:,idx]
                        evals=evals.clip(1e-10) # avoid division by zero
                        evals_sqrt = np.sqrt(evals)
                        # compute euclidan distance in whitented pca space (which is identical to mahalanobis distance)
                        dists = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_rs_centered,evecs)/evals_sqrt, 'euclidean')
                    else:
                        cov=inv(covdiag(dat_cov_tp))  
                        dists=distance.cdist(m_train_tp,X_test_rs,'mahalanobis', VI=cov) # compute distances between all test trials, and average train trials
                else:                    
                    dists=distance.cdist(m_train_tp,X_test_rs,'euclidean')

                distances_temp[:,test_index,irep,tp,:]=dists.reshape(len(angspace_temp),len(test_index),ntps)
                
                if verbose:
                    bar.next()

        distances[ans,:,:,:,:]=np.mean(distances_temp,axis=2,keepdims=False)
    
    distances=distances-np.mean(distances,axis=1,keepdims=True)
    
    distances_flat=np.reshape(distances,(distances.shape[0]*distances.shape[1],distances.shape[2],distances.shape[3],distances.shape[4]),order='F')
    distances_flat=distances_flat-np.mean(distances_flat,axis=0,keepdims=True)
    dec_cos=np.squeeze(-np.mean(np.cos(theta_dists2)*distances_flat,axis=0))

    # order the distances, such that same angle distances are in the middle
    # first, assign each theta to a bin from angspace_full
    temp=np.argmin(abs(circ_dist(angspace_full,theta,all_pairs=True)),axis=1)
    ang_bin_temp=np.tile(angspace_full,(len(theta),1))               
    theta_bins=ang_bin_temp[:,temp][0,:]

    # then, sort the distances based on the distances between the theta_bins
    theta_bin_dists=circ_dist(angspace_full,theta_bins,all_pairs=True)
    theta_bin_dists=theta_bin_dists.transpose()
    theta_bin_dists_abs=np.abs(theta_bin_dists)
    # get index of the minimum distance
    theta_bin_dists_min_ind=np.argmin(theta_bin_dists_abs,axis=0)

    distances_ordered=np.zeros((distances_flat.shape))

    shift_to=np.where(angspace_full==0)[0][0]
    for trl in range(len(theta)):
        distances_ordered[:,trl,:,:] = np.roll(distances_flat[:,trl,:,:], int(shift_to - theta_bin_dists_min_ind[trl]), axis=0)
    
    if verbose:
        bar.finish()
    
    return dec_cos,distances,distances_ordered,angspaces,angspace_full 
#%%  orientation resconstrution using cross-validation, cross-temporal
def dist_theta_kfold_ct_new(data,theta,n_folds=8,n_reps=10,data_trn=None,basis_set=True,angspace='default',ang_steps=4,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True):
    
    if verbose:
        from progress.bar import ChargingBar

    if data_trn is None:
        data_trn=data
        
    if type(angspace)==str:
        if angspace=='default':
            angspace=np.arange(-np.pi,np.pi,np.pi/8) # default is 16 bins
        
    if np.array_equal(angspace,np.unique(theta)):
        ang_steps=1        
                
    bin_width=np.diff(angspace)[0]
    
    x_dummy=np.zeros(len(theta)) # needed for sklearn splitting function
    
    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
            
    ntrls, nchans, ntps=np.shape(X_ts)  
    _,_,ntps_trn=np.shape(X_tr)

    m_temp=np.zeros((len(angspace),nchans,ntps))
    m=m_temp
    
    if dist_metric=='euclidean':
        cov_metric=False 
    
    if verbose:
        bar = ChargingBar('Processing', max=ang_steps*n_reps*n_folds*ntps)
    
    dec_cos=np.empty((ang_steps,ntrls,ntps,ntps))
    distances=np.empty((ang_steps,len(angspace),ntrls,ntps_trn,ntps))
    
    dec_cos[:]=np.NaN
    distances[:]=np.NaN

    angspaces=np.zeros((ang_steps,len(angspace)))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
    
        angspace_temp=angspace+ans*bin_width/ang_steps
        angspaces[ans,:]=angspace_temp

    angspace_full=np.reshape(angspaces,(angspaces.shape[0]*angspaces.shape[1]),order='F')

    theta_dists=circ_dist(angspace_full,theta,all_pairs=True)
    theta_dists=theta_dists.transpose()  

    # add two dimensions to theta_dists
    theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
    theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
    theta_dists2=np.tile(theta_dists_temp,(1,1,ntps_trn,ntps))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
    
        angspace_temp=angspace+ans*bin_width/ang_steps
        
        # convert orientations into bins
        temp=np.argmin(abs(circ_dist(angspace_temp,theta,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta),1))               
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        
        y_subst=temp
        y=bin_orient_rads
                
        rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps) # get splitting object
        
        split_counter=0
        
        distances_temp=np.empty([len(angspace_temp),ntrls,n_reps,ntps_trn,ntps])
        distances_temp[:]=np.NaN
        
        theta_dists=circ_dist(angspace_temp,y,all_pairs=True)
        theta_dists=theta_dists.transpose()                
        
        # theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
        # theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
        # theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
        # theta_dists2=np.tile(theta_dists_temp,(1,1,n_reps,ntps,ntps))
        theta_dists=np.tile(np.expand_dims(theta_dists,axis=-1),(1,1,ntps))
        
        angspace_dist=np.unique(np.round(theta_dists,10))
        if -angspace_dist[-1]==angspace_dist[0]:
            angspace_dist=np.delete(angspace_dist,len(angspace_dist)-1)
               
        for train_index, test_index in rskf.split(X=x_dummy,y=y_subst): # loop over all train/test folds, and repepitions
            
            X_train, X_test = X_tr[train_index,:,:], X_ts[test_index,:,:]
            y_train, y_test = y[train_index], y[test_index]
            y_subst_train, y_subst_test = y_subst[train_index], y_subst[test_index]
                        
            irep=int(np.floor(split_counter/n_folds))
            split_counter=split_counter+1
          
            train_dat_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
            train_dat_cov[:]=np.NaN

            train_dat_res_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
            train_dat_res_cov[:]=np.NaN
            
            if balanced_train_bins: # average over same orientaions of training set, but make sure these averages are based on balanced trials
                count_min=min(np.bincount(y_subst_train))
                for c in range(len(angspace_temp)):
                    temp_dat=X_train[y_train==angspace_temp[c],:,:]
                    ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                    m_temp[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                    if balanced_cov:
                        if residual_cov:
                            train_dat_res_cov = np.append(train_dat_res_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                        train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
            else:
                for c in range(len(angspace_temp)):
                    m_temp[c,:,:]=np.mean(X_train[y_train==angspace_temp[c],:,:],axis=0)
                    
            if basis_set: # smooth the averaged train data with basis set
                m=basis_set_fun(m_temp,angspace_temp,basis_smooth='default')
            else:
                m=m_temp
            
            if not balanced_cov:
                train_dat_cov=X_train # use all train trials if cov is not balanced

            if np.isnan(train_dat_res_cov).all():
                train_dat_res_cov=train_dat_cov  
                               
            # reshape test data for efficient distance computation
            X_test_rs=np.moveaxis(X_test,-1,1)
            X_test_rs=np.reshape(X_test_rs,(len(test_index)*ntps,X_test.shape[1]),order='C')

            for tp in range(ntps_trn):
                m_train_tp=m[:,:,tp]
                       
                if dist_metric=='mahalanobis':
                    dat_cov_tp=train_dat_cov[:,:,tp]
                    dat_cov_res_tp=train_dat_res_cov[:,:,tp]
                    if new_version: # with a lot of dimensions, first performing pca and then using euclidian distance is faster (when using cdist)
                        cov=covdiag(dat_cov_res_tp) # use covariance of the training data for pca
                        train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                        X_test_rs_centered = X_test_rs - train_dat_cov_avg
                        m_train_tp_centered = m_train_tp -train_dat_cov_avg
                        evals,evecs = np.linalg.eigh(cov)
                        idx = evals.argsort()[::-1]
                        evals = evals[idx]
                        evecs = evecs[:,idx]
                        evals=evals.clip(1e-10) # avoid division by zero
                        evals_sqrt = np.sqrt(evals)
                        # compute euclidan distance in whitented pca space (which is identical to mahalanobis distance)
                        dists = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_rs_centered,evecs)/evals_sqrt, 'euclidean')
                    else:
                        cov=inv(covdiag(dat_cov_tp))  
                        dists=distance.cdist(m_train_tp,X_test_rs,'mahalanobis', VI=cov) # compute distances between all test trials, and average train trials
                else:                    
                    dists=distance.cdist(m_train_tp,X_test_rs,'euclidean')

                distances_temp[:,test_index,irep,tp,:]=dists.reshape(len(angspace_temp),len(test_index),ntps)
                
                if verbose:
                    bar.next()

        distances[ans,:,:,:,:]=np.mean(distances_temp,axis=2,keepdims=False)
    
    distances=distances-np.mean(distances,axis=1,keepdims=True)
    
    distances=np.reshape(distances,(distances.shape[0]*distances.shape[1],distances.shape[2],distances.shape[3],distances.shape[4]),order='F')
    distances=distances-np.mean(distances,axis=0,keepdims=True)
    dec_cos=np.squeeze(-np.mean(np.cos(theta_dists2)*distances,axis=0))

    # order the distances, such that same angle distances are in the middle
    # first, assign each theta to a bin from angspace_full
    temp=np.argmin(abs(circ_dist(angspace_full,theta,all_pairs=True)),axis=1)
    ang_bin_temp=np.tile(angspace_full,(len(theta),1))               
    theta_bins=ang_bin_temp[:,temp][0,:]

    # then, sort the distances based on the distances between the theta_bins
    theta_bin_dists=circ_dist(angspace_full,theta_bins,all_pairs=True)
    theta_bin_dists=theta_bin_dists.transpose()
    theta_bin_dists_abs=np.abs(theta_bin_dists)
    # get index of the minimum distance
    theta_bin_dists_min_ind=np.argmin(theta_bin_dists_abs,axis=0)

    distances_ordered=np.zeros((distances.shape))

    shift_to=np.where(angspace_full==0)[0][0]
    for trl in range(len(theta)):
        distances_ordered[:,trl,:,:] = np.roll(distances[:,trl,:,:], int(shift_to - theta_bin_dists_min_ind[trl]), axis=0)
    
    if verbose:
        bar.finish()
    
    return dec_cos,distances_ordered,angspaces,angspace_full 
#%%  orientation resconstruction, no cross-validation, cross-temporal
def dist_theta_ct_new(data,theta,data_trn,theta_trn,n_reps=10,basis_set=True,angspace='default',ang_steps=4,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True):
    
    if verbose:
        from progress.bar import ChargingBar

    if data_trn is None:
        data_trn=data
        
    if type(angspace)==str:
        if angspace=='default':
            angspace=np.arange(-np.pi,np.pi,np.pi/8) # default is 16 bins
        
    if np.array_equal(angspace,np.unique(theta)):
        ang_steps=1        
                
    bin_width=np.diff(angspace)[0]
    
    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
          
    ntrls, nchans_tst, ntps=np.shape(X_ts)
    ntrls_trn, nchans_trn, ntps_trn=np.shape(X_tr) 

    m_temp=np.zeros((len(angspace),nchans_trn,ntps_trn))
    m=m_temp
    
    if verbose:
        bar = ChargingBar('Processing', max=ang_steps*n_reps*ntps_trn)
    
    dec_cos=np.empty((ang_steps,ntrls,ntps_trn,ntps))
    distances=np.empty((ang_steps,len(angspace),ntrls,ntps_trn,ntps))
    
    dec_cos[:]=np.NaN
    distances[:]=np.NaN

    angspaces=np.zeros((ang_steps,len(angspace)))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
        angspace_temp=angspace+ans*bin_width/ang_steps
        angspaces[ans,:]=angspace_temp

    angspace_full=np.reshape(angspaces,(angspaces.shape[0]*angspaces.shape[1]),order='F')

    theta_dists=circ_dist(angspace_full,theta,all_pairs=True)
    theta_dists=theta_dists.transpose()  

    theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
    theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
    theta_dists2=np.tile(theta_dists_temp,(1,1,ntps_trn,ntps))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
        angspace_temp=angspace+ans*bin_width/ang_steps
        
        temp=np.argmin(abs(circ_dist(angspace_temp,theta,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta),1))        
        
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        y_test=bin_orient_rads
        
        distances_temp=np.empty([len(angspace_temp),ntrls,n_reps,ntps_trn,ntps])
        distances_temp[:]=np.NaN

        theta_dists=circ_dist(angspace_temp,y_test,all_pairs=True)
        theta_dists=theta_dists.transpose()                
        
        # theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
        # theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
        # theta_dists2=np.tile(theta_dists_temp,(1,1,n_reps,ntps))
        # theta_dists=np.tile(np.expand_dims(theta_dists,axis=-1),(1,1,ntps))
        
        temp=np.argmin(abs(circ_dist(angspace_temp,theta_trn,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta_trn),1))        
        
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        y_subst_train=temp
        y_train=bin_orient_rads
                             
        angspace_dist=np.unique(np.round(theta_dists,10))
        if -angspace_dist[-1]==angspace_dist[0]:
            angspace_dist=np.delete(angspace_dist,len(angspace_dist)-1)
               
        for irep in range(n_reps):
            train_dat_cov = np.empty((0,X_tr.shape[1],X_tr.shape[2]))
            train_dat_cov[:]=np.NaN

            train_dat_res_cov = np.empty((0,X_tr.shape[1],X_tr.shape[2]))
            train_dat_res_cov[:]=np.NaN
            
            if balanced_train_bins:
                count_min=min(np.bincount(y_subst_train))
                for c in range(len(angspace_temp)):
                    temp_dat=X_tr[y_train==angspace_temp[c],:,:]
                    ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                    m_temp[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                    if balanced_cov:
                        if residual_cov:
                            train_dat_res_cov = np.append(train_dat_res_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                        train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0) 
            else:
                for c in range(len(angspace_temp)):
                    m_temp[c,:,:]=np.mean(X_tr[y_train==angspace_temp[c],:,:],axis=0)
                    
            if basis_set:
                m=basis_set_fun(m_temp,angspace_temp,basis_smooth='default')
            else:
                m=m_temp
        
            if not balanced_cov:
                train_dat_cov=X_tr # use all train trials if cov is not balanced

            if np.isnan(train_dat_res_cov).all():
                train_dat_res_cov=train_dat_cov  
                               
            # reshape test data for efficient distance computation
            X_test_rs=np.moveaxis(X_ts,-1,1)
            X_test_rs=np.reshape(X_test_rs,(X_ts.shape[0]*ntps,X_ts.shape[1]),order='C')

            for tp in range(ntps_trn):
                m_train_tp=m[:,:,tp]
                       
                if dist_metric=='mahalanobis':
                    dat_cov_tp=train_dat_cov[:,:,tp]
                    dat_cov_res_tp=train_dat_res_cov[:,:,tp]
                    if new_version: # with a lot of dimensions, first performing pca and then using euclidian distance is faster (when using cdist)
                        cov=covdiag(dat_cov_res_tp) # use covariance of the training data for pca
                        train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                        X_test_rs_centered = X_test_rs - train_dat_cov_avg
                        m_train_tp_centered = m_train_tp -train_dat_cov_avg
                        evals,evecs = np.linalg.eigh(cov)
                        idx = evals.argsort()[::-1]
                        evals = evals[idx]
                        evecs = evecs[:,idx]
                        evals=evals.clip(1e-10) # avoid division by zero
                        evals_sqrt = np.sqrt(evals)
                        # compute euclidan distance in whitented pca space (which is identical to mahalanobis distance)
                        dists = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_rs_centered,evecs)/evals_sqrt, 'euclidean')
                    else:
                        cov=inv(covdiag(dat_cov_tp))  
                        dists=distance.cdist(m_train_tp,X_test_rs,'mahalanobis', VI=cov) # compute distances between all test trials, and average train trials
                else:                    
                    dists=distance.cdist(m_train_tp,X_test_rs,'euclidean')

                distances_temp[:,:,irep,tp,:]=dists.reshape(len(angspace_temp),X_ts.shape[0],ntps)
                
                if verbose:
                    bar.next()

        distances[ans,:,:,:,:]=np.mean(distances_temp,axis=2,keepdims=False)
    
    distances=distances-np.mean(distances,axis=1,keepdims=True)
    distances=np.reshape(distances,(distances.shape[0]*distances.shape[1],distances.shape[2],distances.shape[3],distances.shape[4]),order='F')
    distances=distances-np.mean(distances,axis=0,keepdims=True)
    dec_cos=np.squeeze(-np.mean(np.cos(theta_dists2)*distances,axis=0))

    # order the distances, such that same angle distances are in the middle
    # first, assign each theta to a bin from angspace_full
    temp=np.argmin(abs(circ_dist(angspace_full,theta,all_pairs=True)),axis=1)
    ang_bin_temp=np.tile(angspace_full,(len(theta),1))               
    theta_bins=ang_bin_temp[:,temp][0,:]

    # then, sort the distances based on the distances between the theta_bins
    theta_bin_dists=circ_dist(angspace_full,theta_bins,all_pairs=True)
    theta_bin_dists=theta_bin_dists.transpose()
    theta_bin_dists_abs=np.abs(theta_bin_dists)
    # get index of the minimum distance
    theta_bin_dists_min_ind=np.argmin(theta_bin_dists_abs,axis=0)

    distances_ordered=np.zeros((distances.shape))

    shift_to=np.where(angspace_full==0)[0][0]
    for trl in range(len(theta)):
        distances_ordered[:,trl,:,:] = np.roll(distances[:,trl,:,:], int(shift_to - theta_bin_dists_min_ind[trl]), axis=0)
    
    if verbose:
        bar.finish()
    
    return dec_cos,distances_ordered,angspaces,angspace_full 
#%%  orientation resconstrution using cross-validation, cross-temporal, compute only decoding accruacy of trial average to reduce memory usage
def dist_theta_kfold_ct_av2(data,theta,n_folds=8,n_reps=10,data_trn=None,basis_set=True,angspace='default',ang_steps=4,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True):
    
    if verbose:
        from progress.bar import ChargingBar

    if data_trn is None:
        data_trn=data
        
    if type(angspace)==str:
        if angspace=='default':
            angspace=np.arange(-np.pi,np.pi,np.pi/8) # default is 16 bins
        
    if np.array_equal(angspace,np.unique(theta)):
        ang_steps=1        
                
    bin_width=np.diff(angspace)[0]
    
    x_dummy=np.zeros(len(theta)) # needed for sklearn splitting function
    
    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
            
    ntrls, nchans, ntps=np.shape(X_ts)  
    _,_,ntps_trn=np.shape(X_tr)

    m_temp=np.zeros((len(angspace),nchans,ntps))
    m=m_temp
    
    if dist_metric=='euclidean':
        cov_metric=False 
    
    if verbose:
        bar = ChargingBar('Processing', max=ang_steps*n_reps*n_folds*ntps)
    
    angspaces=np.zeros((ang_steps,len(angspace)))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
    
        angspace_temp=angspace+ans*bin_width/ang_steps
        angspaces[ans,:]=angspace_temp

    angspace_full=np.reshape(angspaces,(angspaces.shape[0]*angspaces.shape[1]),order='F')

    theta_dists=circ_dist(angspace_full,theta,all_pairs=True)
    theta_dists=theta_dists.transpose()  

    # add two dimensions to theta_dists
    theta_dists_temp=np.expand_dims(theta_dists,axis=-1)

    theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
    theta_dists2=np.tile(theta_dists_temp,(1,1,ntps_trn,ntps))

    dec_cos_av=np.empty((ang_steps,ntps,ntps))
    for ans in range(0,ang_steps): # loop over all desired orientation spaces
    
        angspace_temp=angspace+ans*bin_width/ang_steps

        theta_dists_temp=circ_dist(angspace_temp,theta,all_pairs=True).T
        theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
        theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
        theta_dists2_temp=np.tile(theta_dists_temp,(1,1,ntps_trn,ntps))
        # convert orientations into bins
        temp=np.argmin(abs(circ_dist(angspace_temp,theta,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta),1))               
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        
        y_subst=temp
        y=bin_orient_rads
                
        rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps) # get splitting object
        
        split_counter=0
        
        distances_temp=np.empty([len(angspace_temp),ntrls,n_reps,ntps_trn,ntps])
        distances_temp[:]=np.NaN

        distances_temp2=np.empty([len(angspace_temp),ntrls,ntps_trn,ntps])
        
        theta_dists=circ_dist(angspace_temp,y,all_pairs=True)
        theta_dists=theta_dists.transpose()                
        
        theta_dists=np.tile(np.expand_dims(theta_dists,axis=-1),(1,1,ntps))
        
        angspace_dist=np.unique(np.round(theta_dists,10))
        if -angspace_dist[-1]==angspace_dist[0]:
            angspace_dist=np.delete(angspace_dist,len(angspace_dist)-1)

        dec_cos_ans_av=np.empty((n_reps,ntps,ntps))
        for train_index, test_index in rskf.split(X=x_dummy,y=y_subst): # loop over all train/test folds, and repepitions
            
            X_train, X_test = X_tr[train_index,:,:], X_ts[test_index,:,:]
            y_train, y_test = y[train_index], y[test_index]
            y_subst_train, y_subst_test = y_subst[train_index], y_subst[test_index]
            
            # get repitition number
            irep=int(np.floor(split_counter/n_folds))

            # get current fold
            ifold=split_counter%n_folds
            split_counter=split_counter+1
          
            train_dat_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
            train_dat_cov[:]=np.NaN

            train_dat_res_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
            train_dat_res_cov[:]=np.NaN
            
            if balanced_train_bins: # average over same orientaions of training set, but make sure these averages are based on balanced trials
                count_min=min(np.bincount(y_subst_train))
                for c in range(len(angspace_temp)):
                    temp_dat=X_train[y_train==angspace_temp[c],:,:]
                    ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                    m_temp[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                    if balanced_cov:
                        if residual_cov:
                            train_dat_res_cov = np.append(train_dat_res_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                        train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
            else:
                for c in range(len(angspace_temp)):
                    m_temp[c,:,:]=np.mean(X_train[y_train==angspace_temp[c],:,:],axis=0)
                    
            if basis_set: # smooth the averaged train data with basis set
                m=basis_set_fun(m_temp,angspace_temp,basis_smooth='default')
            else:
                m=m_temp
            
            if not balanced_cov:
                train_dat_cov=X_train # use all train trials if cov is not balanced

            if np.isnan(train_dat_res_cov).all():
                train_dat_res_cov=train_dat_cov  
                               
            # reshape test data for efficient distance computation
            X_test_rs=np.moveaxis(X_test,-1,1)
            X_test_rs=np.reshape(X_test_rs,(len(test_index)*ntps,X_test.shape[1]),order='C')

            for tp in range(ntps_trn):
                m_train_tp=m[:,:,tp]
                       
                if dist_metric=='mahalanobis':
                    dat_cov_tp=train_dat_cov[:,:,tp]
                    dat_cov_res_tp=train_dat_res_cov[:,:,tp]
                    if new_version: # with a lot of dimensions, first performing pca and then using euclidian distance is faster (when using cdist)
                        cov=covdiag(dat_cov_res_tp) # use covariance of the training data for pca
                        train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                        X_test_rs_centered = X_test_rs - train_dat_cov_avg
                        m_train_tp_centered = m_train_tp -train_dat_cov_avg
                        evals,evecs = np.linalg.eigh(cov)
                        idx = evals.argsort()[::-1]
                        evals = evals[idx]
                        evecs = evecs[:,idx]
                        evals=evals.clip(1e-10) # avoid division by zero
                        evals_sqrt = np.sqrt(evals)
                        # compute euclidan distance in whitented pca space (which is identical to mahalanobis distance)
                        dists = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_rs_centered,evecs)/evals_sqrt, 'euclidean')
                    else:
                        cov=inv(covdiag(dat_cov_tp))  
                        dists=distance.cdist(m_train_tp,X_test_rs,'mahalanobis', VI=cov) # compute distances between all test trials, and average train trials
                else:                    
                    dists=distance.cdist(m_train_tp,X_test_rs,'euclidean')

                distances_temp2[:,test_index,tp,:]=dists.reshape(len(angspace_temp),len(test_index),ntps)

                if verbose:
                    bar.next()

            if ifold==n_folds-1:
                dists_mc=distances_temp2-np.mean(distances_temp2,axis=1,keepdims=True)
                dec_cos_ans_av[irep,:,:]=np.mean(np.squeeze(-np.mean(np.cos(theta_dists2_temp)*dists_mc,axis=0)),axis=0)

        dec_cos_av[ans,:,:]=np.mean(dec_cos_ans_av,axis=0)

    dec_cos_av=np.mean(dec_cos_av,axis=0)

    return dec_cos_av
#%%
# without cross-validation; separate training and testing data
def dist_theta(data,theta,data_trn,theta_trn,n_reps=10,basis_set=True,angspace='default',ang_steps=4,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True):
    
    if verbose:
        from progress.bar import ChargingBar

    if type(angspace)==str:
        if angspace=='default':
            angspace=np.arange(-np.pi,np.pi,np.pi/8)
        
    if np.array_equal(angspace,np.unique(theta)):
        ang_steps=1       
                
    bin_width=np.diff(angspace)[0]
    
    X_test=data
    X_train=data_trn    
    if len(X_train.shape)<3:
        X_train=np.expand_dims(X_train,axis=-1)
        
    if len(X_test.shape)<3:
        X_test=np.expand_dims(X_test,axis=-1)
            
    ntrls, nchans, ntps=np.shape(X_test)  

    m_temp=np.zeros((len(angspace),nchans,ntps))
    m=m_temp
    
    if verbose:
        bar = ChargingBar('Processing', max=ntps*ang_steps*n_reps)
    
    dec_cos=np.empty((ang_steps,ntrls,ntps))
    distances=np.empty((ang_steps,len(angspace),ntrls,ntps))
    
    dec_cos[:]=np.NaN
    distances[:]=np.NaN

    angspaces=np.zeros((ang_steps,len(angspace)))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
        angspace_temp=angspace+ans*bin_width/ang_steps
        angspaces[ans,:]=angspace_temp

    angspace_full=np.reshape(angspaces,(angspaces.shape[0]*angspaces.shape[1]),order='F')

    theta_dists=circ_dist(angspace_full,theta,all_pairs=True)
    theta_dists=theta_dists.transpose()  

    theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
    theta_dists2=np.tile(theta_dists_temp,(1,1,ntps))

    for ans in range(0,ang_steps):
        angspace_temp=angspace+ans*bin_width/ang_steps
        
        temp=np.argmin(abs(circ_dist(angspace_temp,theta,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta),1))        
        
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        y_test=bin_orient_rads
        
        distances_temp=np.empty([len(angspace_temp),ntrls,n_reps,ntps])
        distances_temp[:]=np.NaN
        
        theta_dists=circ_dist(angspace_temp,y_test,all_pairs=True);
        theta_dists=theta_dists.transpose()                
        
        temp=np.argmin(abs(circ_dist(angspace_temp,theta_trn,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta_trn),1))        
        
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        y_subst_train=temp
        y_train=bin_orient_rads
                             
        distances_temp=np.empty([len(angspace_temp),ntrls,n_reps,ntps])
        distances_temp[:]=np.NaN
        
        angspace_dist=np.unique(np.round(theta_dists,10))
        if -angspace_dist[-1]==angspace_dist[0]:
            angspace_dist=np.delete(angspace_dist,len(angspace_dist)-1)
                                       
        for irep in range(n_reps):
            train_dat_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
            train_dat_cov[:]=np.NaN
            
            if balanced_train_bins:
                count_min=min(np.bincount(y_subst_train))
                for c in range(len(angspace_temp)):
                    temp_dat=X_train[y_train==angspace_temp[c],:,:]
                    ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                    m_temp[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                    if balanced_cov:
                        if residual_cov: # take the residual, note that this should only be done if the cov data is balanced!
                            train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                        else:
                            train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
            else:
                for c in range(len(angspace_temp)):
                    m_temp[c,:,:]=np.mean(X_train[y_train==angspace_temp[c],:,:],axis=0)
                    
            if basis_set:
                m=basis_set_fun(m_temp,angspace_temp,basis_smooth='default')
            else:
                m=m_temp
        
            if not balanced_cov:
                train_dat_cov=X_train 
                    
            for tp in range(ntps):
                m_train_tp=m[:,:,tp]
                X_test_tp=X_test[:,:,tp]
                
                if dist_metric=='mahalanobis':
                    dat_cov_tp=train_dat_cov[:,:,tp]
                    if new_version: # with a lot of dimensions, first performing pca and then using euclidian distance is faster (when using cdist)
                        cov=covdiag(dat_cov_tp) # use covariance of the training data for pca
                        train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                        X_test_tp_centered = X_test_tp - train_dat_cov_avg
                        m_train_tp_centered = m_train_tp -train_dat_cov_avg
                        evals,evecs = np.linalg.eigh(cov)
                        idx = evals.argsort()[::-1]
                        evals = evals[idx]
                        evecs = evecs[:,idx]
                        # evals=evals.clip(1e-10) # avoid division by zero
                        evals_sqrt = np.sqrt(evals)
                        # compute euclidan distance in whitened pca space (which is identical to mahalanobis distance)
                        distances_temp[:,:,irep,tp] = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_tp_centered,evecs)/evals_sqrt, 'euclidean')
                    else:                                
                        cov=inv(covdiag(dat_cov_tp))  
                        distances_temp[:,:,irep,tp]=distance.cdist(m_train_tp,X_test_tp,'mahalanobis', VI=cov)
                else:                    
                    distances_temp[:,:,irep,tp]=distance.cdist(m_train_tp,X_test_tp,'euclidean')

                if verbose:    
                    bar.next()  

        distances[ans,:,:,:]=np.mean(distances_temp,axis=2,keepdims=False)
    
    distances=distances-np.mean(distances,axis=1,keepdims=True)
    distances_flat=np.reshape(distances,(distances.shape[0]*distances.shape[1],distances.shape[2],distances.shape[3]),order='F')
    distances_flat=distances_flat-np.mean(distances_flat,axis=0,keepdims=True)
    dec_cos=np.squeeze(-np.mean(np.cos(theta_dists2)*distances_flat,axis=0))

    # order the distances, such that same angle distances are in the middle
    # first, assign each theta to a bin from angspace_full
    temp=np.argmin(abs(circ_dist(angspace_full,theta,all_pairs=True)),axis=1)
    ang_bin_temp=np.tile(angspace_full,(len(theta),1))               
    theta_bins=ang_bin_temp[:,temp][0,:]

    # then, sort the distances based on the distances between the theta_bins
    theta_bin_dists=circ_dist(angspace_full,theta_bins,all_pairs=True)
    theta_bin_dists=theta_bin_dists.transpose()
    theta_bin_dists_abs=np.abs(theta_bin_dists)
    # get index of the minimum distance
    theta_bin_dists_min_ind=np.argmin(theta_bin_dists_abs,axis=0)

    distances_ordered=np.zeros((distances_flat.shape))

    shift_to=np.where(angspace_full==0)[0][0]
    for trl in range(len(theta)):
        distances_ordered[:,trl,:] = np.roll(distances_flat[:,trl,:], int(shift_to - theta_bin_dists_min_ind[trl]), axis=0)
    
    if verbose:
        bar.finish()
    
    return dec_cos,distances,distances_ordered,angspaces,angspace_full
#%%  orientation resconstruction, no cross-validation, cross-temporal
def dist_theta_ct(data,theta,data_trn,theta_trn,n_reps=10,basis_set=True,angspace='default',ang_steps=4,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True):
    
    if verbose:
        from progress.bar import ChargingBar

    if data_trn is None:
        data_trn=data
        
    if type(angspace)==str:
        if angspace=='default':
            angspace=np.arange(-np.pi,np.pi,np.pi/8) # default is 16 bins
        
    if np.array_equal(angspace,np.unique(theta)):
        ang_steps=1        
                
    bin_width=np.diff(angspace)[0]
    
    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
          
    ntrls, nchans_tst, ntps=np.shape(X_ts)
    ntrls_trn, nchans_trn, ntps_trn=np.shape(X_tr) 

    m_temp=np.zeros((len(angspace),nchans_trn,ntps_trn))
    m=m_temp
    
    if verbose:
        bar = ChargingBar('Processing', max=ang_steps*n_reps*ntps_trn)
    
    dec_cos=np.empty((ang_steps,ntrls,ntps_trn,ntps))
    distances=np.empty((ang_steps,len(angspace),ntrls,ntps_trn,ntps))
    
    dec_cos[:]=np.NaN
    distances[:]=np.NaN

    angspaces=np.zeros((ang_steps,len(angspace)))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
        angspace_temp=angspace+ans*bin_width/ang_steps
        angspaces[ans,:]=angspace_temp

    angspace_full=np.reshape(angspaces,(angspaces.shape[0]*angspaces.shape[1]),order='F')

    theta_dists=circ_dist(angspace_full,theta,all_pairs=True)
    theta_dists=theta_dists.transpose()  

    theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
    theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
    theta_dists2=np.tile(theta_dists_temp,(1,1,ntps_trn,ntps))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
        angspace_temp=angspace+ans*bin_width/ang_steps
        
        temp=np.argmin(abs(circ_dist(angspace_temp,theta,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta),1))        
        
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        y_test=bin_orient_rads
        
        distances_temp=np.empty([len(angspace_temp),ntrls,n_reps,ntps_trn,ntps])
        distances_temp[:]=np.NaN

        theta_dists=circ_dist(angspace_temp,y_test,all_pairs=True)
        theta_dists=theta_dists.transpose()                
        
        # theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
        # theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
        # theta_dists2=np.tile(theta_dists_temp,(1,1,n_reps,ntps))
        # theta_dists=np.tile(np.expand_dims(theta_dists,axis=-1),(1,1,ntps))
        
        temp=np.argmin(abs(circ_dist(angspace_temp,theta_trn,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta_trn),1))        
        
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        y_subst_train=temp
        y_train=bin_orient_rads
                             
        angspace_dist=np.unique(np.round(theta_dists,10))
        if -angspace_dist[-1]==angspace_dist[0]:
            angspace_dist=np.delete(angspace_dist,len(angspace_dist)-1)
               
        for irep in range(n_reps):
            train_dat_cov = np.empty((0,X_tr.shape[1],X_tr.shape[2]))
            train_dat_cov[:]=np.NaN
            
            if balanced_train_bins: # average over same orientaions of training set, but make sure these averages are based on balanced trials
                count_min=min(np.bincount(y_subst_train))
                for c in range(len(angspace_temp)):
                    temp_dat=X_tr[y_train==angspace_temp[c],:,:]
                    ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                    m_temp[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                    if balanced_cov: # if desired, the data used for the covariance can also be balanced
                        if residual_cov: # take the residual, note that this should only be done if the cov data is balanced!
                            train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                        else:
                            train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
            else:
                for c in range(len(angspace_temp)):
                    m_temp[c,:,:]=np.mean(X_tr[y_train==angspace_temp[c],:,:],axis=0)
                    
            if basis_set: # smooth the averaged train data with basis set
                m=basis_set_fun(m_temp,angspace_temp,basis_smooth='default')
            else:
                m=m_temp
                                       
            if not balanced_cov:
                train_dat_cov=X_tr # use all train trials if cov is not balanced
                               
            # reshape test data for efficient distance computation
            X_test_rs=np.moveaxis(X_ts,-1,1)
            X_test_rs=np.reshape(X_test_rs,(X_ts.shape[0]*ntps,X_ts.shape[1]),order='C')

            for tp in range(ntps_trn):
                m_train_tp=m[:,:,tp]
                       
                if dist_metric=='mahalanobis':
                    dat_cov_tp=train_dat_cov[:,:,tp]
                    if new_version: # with a lot of dimensions, first performing pca and then using euclidian distance is faster (when using cdist)
                        cov=covdiag(dat_cov_tp) # use covariance of the training data for pca
                        train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                        X_test_rs_centered = X_test_rs - train_dat_cov_avg
                        m_train_tp_centered = m_train_tp -train_dat_cov_avg
                        evals,evecs = np.linalg.eigh(cov)
                        idx = evals.argsort()[::-1]
                        evals = evals[idx]
                        evecs = evecs[:,idx]
                        evals=evals.clip(1e-10) # avoid division by zero
                        evals_sqrt = np.sqrt(evals)
                        # compute euclidan distance in whitented pca space (which is identical to mahalanobis distance)
                        dists = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_rs_centered,evecs)/evals_sqrt, 'euclidean')
                    else:
                        cov=inv(covdiag(dat_cov_tp))  
                        dists=distance.cdist(m_train_tp,X_test_rs,'mahalanobis', VI=cov) # compute distances between all test trials, and average train trials
                else:                    
                    dists=distance.cdist(m_train_tp,X_test_rs,'euclidean')

                distances_temp[:,:,irep,tp,:]=dists.reshape(len(angspace_temp),X_ts.shape[0],ntps)
                
                if verbose:
                    bar.next()

        distances[ans,:,:,:,:]=np.mean(distances_temp,axis=2,keepdims=False)
    
    distances=distances-np.mean(distances,axis=1,keepdims=True)
    distances_flat=np.reshape(distances,(distances.shape[0]*distances.shape[1],distances.shape[2],distances.shape[3],distances.shape[4]),order='F')
    distances_flat=distances_flat-np.mean(distances_flat,axis=0,keepdims=True)
    dec_cos=np.squeeze(-np.mean(np.cos(theta_dists2)*distances_flat,axis=0))

    # order the distances, such that same angle distances are in the middle
    # first, assign each theta to a bin from angspace_full
    temp=np.argmin(abs(circ_dist(angspace_full,theta,all_pairs=True)),axis=1)
    ang_bin_temp=np.tile(angspace_full,(len(theta),1))               
    theta_bins=ang_bin_temp[:,temp][0,:]

    # then, sort the distances based on the distances between the theta_bins
    theta_bin_dists=circ_dist(angspace_full,theta_bins,all_pairs=True)
    theta_bin_dists=theta_bin_dists.transpose()
    theta_bin_dists_abs=np.abs(theta_bin_dists)
    # get index of the minimum distance
    theta_bin_dists_min_ind=np.argmin(theta_bin_dists_abs,axis=0)

    distances_ordered=np.zeros((distances_flat.shape))

    shift_to=np.where(angspace_full==0)[0][0]
    for trl in range(len(theta)):
        distances_ordered[:,trl,:,:] = np.roll(distances_flat[:,trl,:,:], int(shift_to - theta_bin_dists_min_ind[trl]), axis=0)
    
    if verbose:
        bar.finish()
    
    return dec_cos,distances,distances_ordered,angspaces,angspace_full 
#%%
def dist_theta_ct2(data,theta,theta2,data_trn,theta_trn,n_reps=10,basis_set=True,angspace='default',ang_steps=4,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True):
    
    if verbose:
        from progress.bar import ChargingBar

    if data_trn is None:
        data_trn=data
        
    if type(angspace)==str:
        if angspace=='default':
            angspace=np.arange(-np.pi,np.pi,np.pi/8) # default is 16 bins
        
    if np.array_equal(angspace,np.unique(theta)):
        ang_steps=1        
                
    bin_width=np.diff(angspace)[0]
    
    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
          
    ntrls, nchans_tst, ntps=np.shape(X_ts)
    ntrls_trn, nchans_trn, ntps_trn=np.shape(X_tr) 

    m_temp=np.zeros((len(angspace),nchans_trn,ntps_trn))
    m=m_temp
    
    if verbose:
        bar = ChargingBar('Processing', max=ang_steps*n_reps*ntps_trn)
    
    dec_cos=np.empty((ang_steps,ntrls,ntps_trn,ntps))
    distances=np.empty((ang_steps,len(angspace),ntrls,ntps_trn,ntps))
    
    dec_cos[:]=np.NaN
    distances[:]=np.NaN

    angspaces=np.zeros((ang_steps,len(angspace)))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
        angspace_temp=angspace+ans*bin_width/ang_steps
        angspaces[ans,:]=angspace_temp

    angspace_full=np.reshape(angspaces,(angspaces.shape[0]*angspaces.shape[1]),order='F')

    theta_dists=circ_dist(angspace_full,theta,all_pairs=True)
    theta_dists=theta_dists.transpose()  

    theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
    theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
    theta_dists2=np.tile(theta_dists_temp,(1,1,ntps_trn,ntps))

    theta_dists_2=circ_dist(angspace_full,theta,all_pairs=True)
    theta_dists_2=theta_dists_2.transpose()  

    theta_dists_temp_2=np.expand_dims(theta_dists_2,axis=-1)
    theta_dists_temp_2=np.expand_dims(theta_dists_temp_2,axis=-1)
    theta_dists2_2=np.tile(theta_dists_temp_2,(1,1,ntps_trn,ntps))

    for ans in range(0,ang_steps): # loop over all desired orientation spaces
        angspace_temp=angspace+ans*bin_width/ang_steps
        
        temp=np.argmin(abs(circ_dist(angspace_temp,theta,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta),1))        
        
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        y_test=bin_orient_rads
        
        distances_temp=np.empty([len(angspace_temp),ntrls,n_reps,ntps_trn,ntps])
        distances_temp[:]=np.NaN

        theta_dists=circ_dist(angspace_temp,y_test,all_pairs=True)
        theta_dists=theta_dists.transpose()                
        
        # theta_dists_temp=np.expand_dims(theta_dists,axis=-1)
        # theta_dists_temp=np.expand_dims(theta_dists_temp,axis=-1)
        # theta_dists2=np.tile(theta_dists_temp,(1,1,n_reps,ntps))
        # theta_dists=np.tile(np.expand_dims(theta_dists,axis=-1),(1,1,ntps))
        
        temp=np.argmin(abs(circ_dist(angspace_temp,theta_trn,all_pairs=True)),axis=1)
        ang_bin_temp=np.tile(angspace_temp,(len(theta_trn),1))        
        
        bin_orient_rads=ang_bin_temp[:,temp][0,:]
        y_subst_train=temp
        y_train=bin_orient_rads
                             
        angspace_dist=np.unique(np.round(theta_dists,10))
        if -angspace_dist[-1]==angspace_dist[0]:
            angspace_dist=np.delete(angspace_dist,len(angspace_dist)-1)
               
        for irep in range(n_reps):
            train_dat_cov = np.empty((0,X_tr.shape[1],X_tr.shape[2]))
            train_dat_cov[:]=np.NaN

            train_dat_res_cov = np.empty((0,X_tr.shape[1],X_tr.shape[2]))
            train_dat_res_cov[:]=np.NaN
            
            if balanced_train_bins:
                count_min=min(np.bincount(y_subst_train))
                for c in range(len(angspace_temp)):
                    temp_dat=X_tr[y_train==angspace_temp[c],:,:]
                    ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                    m_temp[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                    if balanced_cov:
                        if residual_cov:
                            train_dat_res_cov = np.append(train_dat_res_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                        train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0) 
            else:
                for c in range(len(angspace_temp)):
                    m_temp[c,:,:]=np.mean(X_tr[y_train==angspace_temp[c],:,:],axis=0)
                    
            if basis_set:
                m=basis_set_fun(m_temp,angspace_temp,basis_smooth='default')
            else:
                m=m_temp
        
            if not balanced_cov:
                train_dat_cov=X_tr # use all train trials if cov is not balanced

            if np.isnan(train_dat_res_cov).all():
                train_dat_res_cov=train_dat_cov  
                               
            # reshape test data for efficient distance computation
            X_test_rs=np.moveaxis(X_ts,-1,1)
            X_test_rs=np.reshape(X_test_rs,(X_ts.shape[0]*ntps,X_ts.shape[1]),order='C')

            for tp in range(ntps_trn):
                m_train_tp=m[:,:,tp]
                       
                if dist_metric=='mahalanobis':
                    dat_cov_tp=train_dat_cov[:,:,tp]
                    dat_cov_res_tp=train_dat_res_cov[:,:,tp]
                    if new_version: # with a lot of dimensions, first performing pca and then using euclidian distance is faster (when using cdist)
                        cov=covdiag(dat_cov_res_tp) # use covariance of the training data for pca
                        train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                        X_test_rs_centered = X_test_rs - train_dat_cov_avg
                        m_train_tp_centered = m_train_tp -train_dat_cov_avg
                        evals,evecs = np.linalg.eigh(cov)
                        idx = evals.argsort()[::-1]
                        evals = evals[idx]
                        evecs = evecs[:,idx]
                        evals=evals.clip(1e-10) # avoid division by zero
                        evals_sqrt = np.sqrt(evals)
                        # compute euclidan distance in whitented pca space (which is identical to mahalanobis distance)
                        dists = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_rs_centered,evecs)/evals_sqrt, 'euclidean')
                    else:
                        cov=inv(covdiag(dat_cov_tp))  
                        dists=distance.cdist(m_train_tp,X_test_rs,'mahalanobis', VI=cov) # compute distances between all test trials, and average train trials
                else:                    
                    dists=distance.cdist(m_train_tp,X_test_rs,'euclidean')

                distances_temp[:,:,irep,tp,:]=dists.reshape(len(angspace_temp),X_ts.shape[0],ntps)
                
                if verbose:
                    bar.next()

        distances[ans,:,:,:,:]=np.mean(distances_temp,axis=2,keepdims=False)
    
    distances=distances-np.mean(distances,axis=1,keepdims=True)
    distances_flat=np.reshape(distances,(distances.shape[0]*distances.shape[1],distances.shape[2],distances.shape[3],distances.shape[4]),order='F')
    distances_flat=distances_flat-np.mean(distances_flat,axis=0,keepdims=True)
    dec_cos=np.squeeze(-np.mean(np.cos(theta_dists2)*distances_flat,axis=0))
    dec_cos2=np.squeeze(-np.mean(np.cos(theta_dists2_2)*distances_flat,axis=0))

    # order the distances, such that same angle distances are in the middle
    # first, assign each theta to a bin from angspace_full
    temp=np.argmin(abs(circ_dist(angspace_full,theta,all_pairs=True)),axis=1)
    ang_bin_temp=np.tile(angspace_full,(len(theta),1))               
    theta_bins=ang_bin_temp[:,temp][0,:]

    # then, sort the distances based on the distances between the theta_bins
    theta_bin_dists=circ_dist(angspace_full,theta_bins,all_pairs=True)
    theta_bin_dists=theta_bin_dists.transpose()
    theta_bin_dists_abs=np.abs(theta_bin_dists)
    # get index of the minimum distance
    theta_bin_dists_min_ind=np.argmin(theta_bin_dists_abs,axis=0)

    distances_ordered=np.zeros((distances_flat.shape))

    shift_to=np.where(angspace_full==0)[0][0]
    for trl in range(len(theta)):
        distances_ordered[:,trl,:,:] = np.roll(distances_flat[:,trl,:,:], int(shift_to - theta_bin_dists_min_ind[trl]), axis=0)
    
    if verbose:
        bar.finish()
    
    return dec_cos,dec_cos2,distances,distances_ordered,angspaces,angspace_full 
#%% categorical decoding using cross-validation   
def dist_nominal_kfold(data,conditions,n_folds=8,n_reps=10,data_trn=None,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True):
    
    if verbose:
        from progress.bar import ChargingBar
    
    if data_trn is None:
        data_trn=data
        
    x_dummy=np.zeros(len(conditions))
    u_conds=np.unique(conditions)
    
    # convert conditions to integers, in case they aren't
    y_subst=np.zeros(conditions.shape)       
    for c in range(len(u_conds)):
        y_subst[conditions==u_conds[c]]=c
    y_subst = y_subst.astype(int)
    u_conds=np.unique(y_subst)
    
    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
                    
    ntrls, nchans, ntps=np.shape(X_ts)
    
    m=np.zeros((len(u_conds),nchans,ntps))   
    
    if verbose:
        bar = ChargingBar('Processing', max=ntps*n_reps*n_folds)
    
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps)
        
    distances_temp=np.empty([len(u_conds),ntrls,n_reps,ntps])
    distances_temp[:]=np.NaN  
    
    split_counter=0  
    
    y_subst=np.squeeze(y_subst)
    for train_index, test_index in rskf.split(X=x_dummy,y=y_subst):
                
        X_train, X_test = X_tr[train_index,:,:], X_ts[test_index,:,:]
        y_train, y_test = y_subst[train_index], y_subst[test_index]
                    
        irep=int(np.floor(split_counter/n_folds))
        split_counter=split_counter+1
      
        train_dat_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
        train_dat_cov[:]=np.NaN

        train_dat_res_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
        train_dat_res_cov[:]=np.NaN
        
        if balanced_train_bins:
            count_min=min(np.bincount(y_train))
            for c in range(len(u_conds)):
                temp_dat=X_train[y_train==u_conds[c],:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                if balanced_cov:
                    if residual_cov:
                        train_dat_res_cov = np.append(train_dat_res_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                    train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
        else:
            for c in range(len(u_conds)):
                m[c,:,:]=np.mean(X_train[y_train==u_conds[c],:,:],axis=0)         
        
        if not balanced_cov:
            train_dat_cov=X_train

        if np.isnan(train_dat_res_cov).all():
            train_dat_res_cov=train_dat_cov
            
        for tp in range(ntps):
            m_train_tp=m[:,:,tp]
            X_test_tp=X_test[:,:,tp]
            
            if dist_metric=='mahalanobis':
                dat_cov_tp=train_dat_cov[:,:,tp]
                dat_cov_res_tp=train_dat_res_cov[:,:,tp]
                if new_version: # euclidian in pca space (same as mahalanobis distance) is faster for high-dimensional data
                    cov=covdiag(dat_cov_res_tp) # use covariance of the training data for pca
                    train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                    X_test_tp_centered = X_test_tp - train_dat_cov_avg
                    m_train_tp_centered = m_train_tp -train_dat_cov_avg
                    evals,evecs = np.linalg.eigh(cov)
                    idx = evals.argsort()[::-1]
                    evals = evals[idx]
                    evecs = evecs[:,idx]
                    evals=evals.clip(1e-10) # avoid division by zero
                    evals_sqrt = np.sqrt(evals)
                    # compute euclidan distance in whitented pca space (which is identical to mahalanobis distance)
                    distances_temp[:,test_index,irep,tp] = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_tp_centered,evecs)/evals_sqrt, 'euclidean')
                else:                                 
                    cov=inv(covdiag(dat_cov_tp))                                     
                    distances_temp[:,test_index,irep,tp]=distance.cdist(m_train_tp,X_test_tp,'mahalanobis', VI=cov)
            else:                    
                distances_temp[:,test_index,irep,tp]=distance.cdist(m_train_tp,X_test_tp,'euclidean')
            if verbose:
                bar.next()

    distances=np.mean(distances_temp,axis=2,keepdims=False)
    
    pred_cond=np.argmin(distances,axis=0)
    temp=np.transpose(np.tile(y_subst,(pred_cond.shape[1],1)))
    dec_acc=pred_cond==temp
    
    distance_difference=np.zeros([ntrls,ntps])
    
    for cond in u_conds:
        temp1=distances[np.setdiff1d(u_conds,cond),:,:]
        temp2=temp1[:,y_subst==cond,:]
        distance_difference[y_subst==cond,:]=np.mean(temp2,axis=0,keepdims=False)-distances[cond,y_subst==cond,:]    
    
    if verbose:
        bar.finish()
    return distance_difference,distances,dec_acc,pred_cond
#%%  cross-temporal   
def dist_nominal_kfold_ct(data,conditions,n_folds=8,n_reps=10,data_trn=None,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True):
    
    if verbose:
        from progress.bar import ChargingBar

    if data_trn is None:
        data_trn=data
        
    x_dummy=np.zeros(len(conditions))
    u_conds=np.unique(conditions)
    
    # convert conditions to integers, in case they aren't
    y_subst=np.zeros(conditions.shape)       
    for c in range(len(u_conds)):
        y_subst[conditions==u_conds[c]]=c
    y_subst = y_subst.astype(int)
    u_conds=np.unique(y_subst)
    
    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
          
    ntrls, nchans, ntps=np.shape(X_ts)
    _,_,ntps_trn=np.shape(X_tr)
    
    m=np.zeros((len(u_conds),nchans,ntps_trn))   
    
    if verbose:
        bar = ChargingBar('Processing', max=ntps_trn*n_reps*n_folds)
    
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps)
        
    distances_temp=np.empty([len(u_conds),ntrls,n_reps,ntps_trn,ntps])
    distances_temp[:]=np.NaN  
    
    split_counter=0  
       
    y_subst=np.squeeze(y_subst)
    for train_index, test_index in rskf.split(X=x_dummy,y=y_subst):
                
        X_train, X_test = X_tr[train_index,:,:], X_ts[test_index,:,:]
        y_train, y_test = y_subst[train_index], y_subst[test_index]
                    
        irep=int(np.floor(split_counter/n_folds))
        split_counter=split_counter+1
      
        train_dat_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
        train_dat_cov[:]=np.NaN

        train_dat_res_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
        train_dat_res_cov[:]=np.NaN
        
        if balanced_train_bins:
            count_min=min(np.bincount(y_train))
            for c in range(len(u_conds)):
                temp_dat=X_train[y_train==u_conds[c],:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                if balanced_cov:
                    if residual_cov:
                        train_dat_res_cov = np.append(train_dat_res_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                    train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
        else:
            for c in range(len(u_conds)):
                m[c,:,:]=np.mean(X_train[y_train==u_conds[c],:,:],axis=0)         
        
        if not balanced_cov:
            train_dat_cov=X_train

        if np.isnan(train_dat_res_cov).all():
            train_dat_res_cov=train_dat_cov                       
        
        # reshape test data for efficient distance computation
        X_test_rs=np.moveaxis(X_test,-1,1)
        X_test_rs=np.reshape(X_test_rs,(len(test_index)*ntps,X_test.shape[1]),order='C')

        for tp in range(ntps_trn):
            m_train_tp=m[:,:,tp]
            
            if dist_metric=='mahalanobis':
                dat_cov_tp=train_dat_cov[:,:,tp]
                dat_cov_res_tp=train_dat_res_cov[:,:,tp]
                if new_version: # with a lot of dimensions, first performing pca and then using euclidian distance is faster (when using cdist)
                    cov=covdiag(dat_cov_res_tp) # use covariance of the training data for pca
                    train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                    X_test_rs_centered = X_test_rs - train_dat_cov_avg
                    m_train_tp_centered = m_train_tp -train_dat_cov_avg
                    evals,evecs = np.linalg.eigh(cov)
                    idx = evals.argsort()[::-1]
                    evals = evals[idx]
                    evecs = evecs[:,idx]
                    evals=evals.clip(1e-10) # avoid division by zero
                    evals_sqrt = np.sqrt(evals)
                    # compute euclidan distance in whitented pca space (which is identical to mahalanobis distance)
                    dists = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_rs_centered,evecs)/evals_sqrt, 'euclidean')
                else:
                    cov=inv(covdiag(dat_cov_tp))
                    dists=distance.cdist(m_train_tp,X_test_rs,'mahalanobis', VI=cov) # compute distances between all test trials, and average train trials
            else:                    
                dists=distance.cdist(m_train_tp,X_test_rs,'euclidean')
            
            distances_temp[:,test_index,irep,tp,:]=dists.reshape(len(u_conds),len(test_index),ntps)

            if verbose:
                bar.next()

    distances=np.mean(distances_temp,axis=2,keepdims=False)
    
    pred_cond=np.argmin(distances,axis=0)
    temp=np.transpose(np.tile(y_subst,(pred_cond.shape[2],pred_cond.shape[1],1)))
    dec_acc=pred_cond==temp
    
    distance_difference=np.zeros([ntrls,ntps_trn,ntps])
    
    for cond in u_conds:
        temp1=distances[np.setdiff1d(u_conds,cond),:,:,:]
        temp2=temp1[:,y_subst==cond,:,:]
        distance_difference[y_subst==cond,:,:]=np.mean(temp2,axis=0,keepdims=False)-distances[cond,y_subst==cond,:,:]    
    
    if verbose:
        bar.finish()
    return distance_difference,distances,dec_acc,pred_cond
#%% categorical decoding, with separate training and testing data  
def dist_nominal(data,conditions,data_trn,conditions_trn,n_reps=10,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True):
    
    if verbose:
        from progress.bar import ChargingBar
        
    u_conds_test=np.unique(conditions)
    u_conds_train=np.unique(conditions_trn)
    
    # convert conditions to integers, in case they aren't
    y_test=np.zeros(conditions.shape)    
    y_train=np.zeros(conditions_trn.shape)       
    for c in range(len(u_conds_test)):
        y_test[conditions==u_conds_test[c]]=c
    for c in range(len(u_conds_train)):
        y_train[conditions_trn==u_conds_train[c]]=c

    y_test=y_test.astype(int)
    y_train=y_train.astype(int)

    u_conds_test=np.unique(y_test)
    u_conds_train=np.unique(y_train)
    
    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
                    
    ntrls_tst, nchans_tst, ntps_tst=np.shape(X_ts)
    ntrls_trn, nchans_trn, ntps_trn=np.shape(X_tr)
    
    m=np.zeros((len(u_conds_train),nchans_trn,ntps_trn))   
    
    if verbose:
        bar = ChargingBar('Processing', max=ntps_trn*n_reps)
            
    distances_temp=np.empty([len(u_conds_test),ntrls_tst,n_reps,ntps_tst])
    distances_temp[:]=np.NaN                 

    X_train, X_test = X_tr, X_ts
    # y_train=conditions_trn
    # y_test=conditions            
      
    for irep in range(n_reps):
        train_dat_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
        train_dat_cov[:]=np.NaN

        train_dat_res_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
        train_dat_res_cov[:]=np.NaN
        
        if balanced_train_bins:
            count_min=min(np.bincount(y_train))
            for c in range(len(u_conds_train)):
                temp_dat=X_train[y_train==u_conds_train[c],:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                if balanced_cov:
                    if residual_cov:
                        train_dat_res_cov = np.append(train_dat_res_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                    train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0) 
        else:
            for c in range(len(u_conds_train)):
                m[c,:,:]=np.mean(X_train[y_train==u_conds_train[c],:,:],axis=0)         
        
        if not balanced_cov:
            train_dat_cov=X_train

        if np.isnan(train_dat_res_cov).all():
            train_dat_res_cov=train_dat_cov
            
        for tp in range(ntps_trn):
            m_train_tp=m[:,:,tp]
            X_test_tp=X_test[:,:,tp]
            
            if dist_metric=='mahalanobis':
                dat_cov_tp=train_dat_cov[:,:,tp]
                dat_cov_res_tp=train_dat_res_cov[:,:,tp]
                if new_version: # with a lot of dimensions, first performing pca and then using euclidian distance is faster (when using cdist)
                    cov=covdiag(dat_cov_res_tp) # use covariance of the training data for pca
                    train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                    X_test_tp_centered = X_test_tp - train_dat_cov_avg
                    m_train_tp_centered = m_train_tp -train_dat_cov_avg
                    evals,evecs = np.linalg.eigh(cov)
                    idx = evals.argsort()[::-1]
                    evals = evals[idx]
                    evecs = evecs[:,idx]
                    evals=evals.clip(1e-10) # avoid division by zero
                    evals_sqrt = np.sqrt(evals)
                    # compute euclidan distance in whitented pca space (which is identical to mahalanobis distance)
                    distances_temp[:,:,irep,tp] = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_tp_centered,evecs)/evals_sqrt, 'euclidean')
                else:                                 
                    cov=inv(covdiag(dat_cov_tp))                                     
                    distances_temp[:,:,irep,tp]=distance.cdist(m_train_tp,X_test_tp,'mahalanobis', VI=cov)
            else:                    
                distances_temp[:,:,irep,tp]=distance.cdist(m_train_tp,X_test_tp,'euclidean')
            if verbose:
                bar.next()

    distances=np.mean(distances_temp,axis=2,keepdims=False)
    
    pred_cond=np.argmin(distances,axis=0)
    temp=np.transpose(np.tile(y_test,(pred_cond.shape[1],1)))
    dec_acc=pred_cond==temp
    
    distance_difference=np.zeros([ntrls_tst,ntps_tst])
    
    for cond in u_conds_test:
        temp1=distances[np.setdiff1d(u_conds_test,cond),:,:]
        temp2=temp1[:,y_test==cond,:]
        distance_difference[y_test==cond,:]=np.mean(temp2,axis=0,keepdims=False)-distances[cond,y_test==cond,:]    
    
    if verbose:
        bar.finish()
    return distance_difference,distances,dec_acc,pred_cond
#%%  cross-temporal, with separate training and testing data, no cross-validation   
def dist_nominal_ct(data,conditions,data_trn,conditions_trn,n_reps=10,balanced_train_bins=True,balanced_cov=False,residual_cov=False,dist_metric='mahalanobis',verbose=True,new_version=True):
    
    if verbose:
        from progress.bar import ChargingBar

    
    u_conds_test=np.unique(conditions)
    u_conds_train=np.unique(conditions_trn)
    
    # convert conditions to integers, in case they aren't
    y_test=np.zeros(conditions.shape)    
    y_train=np.zeros(conditions_trn.shape)       
    for c in range(len(u_conds_test)):
        y_test[conditions==u_conds_test[c]]=c
    for c in range(len(u_conds_train)):
        y_train[conditions_trn==u_conds_train[c]]=c
    y_test=y_test.astype(int)
    y_train=y_train.astype(int)
    
    u_conds=np.unique(y_train)

    X_ts=data
    X_tr=data_trn    
    if len(X_tr.shape)<3:
        X_tr=np.expand_dims(X_tr,axis=-1)
        
    if len(X_ts.shape)<3:
        X_ts=np.expand_dims(X_ts,axis=-1)
          
    ntrls_tst, nchans_tst, ntps_tst=np.shape(X_ts)
    ntrls_trn, nchans_trn, ntps_trn=np.shape(X_tr)
    
    m=np.zeros((len(u_conds),nchans_trn,ntps_trn))   
    
    if verbose:
        bar = ChargingBar('Processing', max=ntps_trn*n_reps)
        
    distances_temp=np.empty([len(u_conds),ntrls_tst,n_reps,ntps_trn,ntps_tst])
    distances_temp[:]=np.NaN  
    
    X_train, X_test = X_tr, X_ts
    # y_train=conditions_trn
    # y_test=conditions     
       
    y_test=np.squeeze(y_test)
    for irep in range(n_reps):
                
        train_dat_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
        train_dat_cov[:]=np.NaN

        train_dat_res_cov = np.empty((0,X_train.shape[1],X_train.shape[2]))
        train_dat_res_cov[:]=np.NaN
        
        if balanced_train_bins:
            count_min=min(np.bincount(y_train))
            for c in range(len(u_conds)):
                temp_dat=X_train[y_train==u_conds[c],:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m[c,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                if balanced_cov:
                    if residual_cov:
                        train_dat_res_cov = np.append(train_dat_res_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                    train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
        else:
            for c in range(len(u_conds)):
                m[c,:,:]=np.mean(X_train[y_train==u_conds[c],:,:],axis=0)         
        
        if not balanced_cov:
            train_dat_cov=X_train

        if np.isnan(train_dat_res_cov).all():
            train_dat_res_cov=train_dat_cov   
        
        # reshape test data for efficient distance computation
        X_test_rs=np.moveaxis(X_test,-1,1)
        X_test_rs=np.reshape(X_test_rs,(ntrls_tst*ntps_tst,X_test.shape[1]),order='C')

        for tp in range(ntps_trn):
            m_train_tp=m[:,:,tp]
            
            if dist_metric=='mahalanobis':
                dat_cov_tp=train_dat_cov[:,:,tp]
                dat_cov_res_tp=train_dat_res_cov[:,:,tp]
                if new_version: # with a lot of dimensions, first performing pca and then using euclidian distance is faster (when using cdist)
                    cov=covdiag(dat_cov_res_tp) # use covariance of the training data for pca
                    train_dat_cov_avg = dat_cov_tp.mean(axis=0)
                    X_test_rs_centered = X_test_rs - train_dat_cov_avg
                    m_train_tp_centered = m_train_tp -train_dat_cov_avg
                    evals,evecs = np.linalg.eigh(cov)
                    idx = evals.argsort()[::-1]
                    evals = evals[idx]
                    evecs = evecs[:,idx]
                    evals=evals.clip(1e-10) # avoid division by zero
                    evals_sqrt = np.sqrt(evals)
                        # compute euclidan distance in whitented pca space (which is identical to mahalanobis distance)
                    dists = distance.cdist(np.dot(m_train_tp_centered,evecs)/evals_sqrt, np.dot(X_test_rs_centered,evecs)/evals_sqrt, 'euclidean')
                else:
                    cov=inv(covdiag(dat_cov_tp))
                    dists=distance.cdist(m_train_tp,X_test_rs,'mahalanobis', VI=cov) # compute distances between all test trials, and average train trials
            else:                    
                dists=distance.cdist(m_train_tp,X_test_rs,'euclidean')
            
            distances_temp[:,:,irep,tp,:]=dists.reshape(len(u_conds),ntrls_tst,ntps_tst)

            if verbose:
                bar.next()

    distances=np.mean(distances_temp,axis=2,keepdims=False)
    
    pred_cond=np.argmin(distances,axis=0)
    temp=np.transpose(np.tile(y_test,(pred_cond.shape[2],pred_cond.shape[1],1)))
    dec_acc=pred_cond==temp
    
    distance_difference=np.zeros([ntrls_tst,ntps_trn,ntps_tst])
    
    for cond in u_conds:
        temp1=distances[np.setdiff1d(u_conds,cond),:,:,:]
        temp2=temp1[:,y_test==cond,:,:]
        distance_difference[y_test==cond,:,:]=np.mean(temp2,axis=0,keepdims=False)-distances[cond,y_test==cond,:,:]    
    
    if verbose:
        bar.finish()
    return distance_difference,distances,dec_acc,pred_cond