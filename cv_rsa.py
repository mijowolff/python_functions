# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:15:22 2022

@author: mijow

cross-validated RSA using mahalanobis distance
"""
# import os
# os.environ["OPENBLAS_NUM_THREADS"] = '1'
from sklearn.model_selection import RepeatedStratifiedKFold,RepeatedKFold
from scipy.stats import zscore
import numpy as np
import random
from numpy.linalg import pinv,inv
from progress.bar import ChargingBar
from scipy.stats import pearsonr,spearmanr 
import pandas as pd
#%% covariance with shrinkage estimator
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
    shrinkage=max(0,min(1,r2/d))
    sigma=shrinkage*prior+(1-shrinkage)*sample
    
    return sigma
#%%
def mahal_CV_RSA(data,conditions,n_folds=8,n_reps=100,data_trn=None,cov_metric='covdiag',cov_tp=True,balanced_train_dat=True,balanced_test_dat=True,
                 balanced_cov=True,residual_cov=False,null_decoding=False,average=True):
    
    if len(data.shape)<3:
        data=np.expand_dims(data,axis=-1)
    if data_trn is None:
        data_trn=data
    
    if len(data_trn.shape)<3:
        data_trn=np.expand_dims(data_trn,axis=-1)
    
        
    ntrls, nchans, ntps=np.shape(data)  
    
    # get all unique conditions combinations
    cond_combs= np.unique(conditions, axis=0)  
    _, conds_id= np.unique(np.concatenate((conditions,cond_combs)),axis=0,return_inverse=True)
    conds_id=conds_id[:ntrls]
    u_conds=np.unique(conds_id)
    n_conds=len(u_conds)
    
    #%%
    RDM=np.zeros((n_reps,n_folds,n_conds,n_conds,ntps))
    RDM[:]=np.nan
    
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps)
    
    x_dummy=np.zeros(ntrls)
    
    split_counter=0
    
    # bar = ChargingBar('Processing', max=ntps*n_reps*n_folds*n_conds)
    bar = ChargingBar('Processing', max=n_reps*n_folds)
    
    for train_index, test_index in rskf.split(X=x_dummy,y=conds_id):
        
        irep=int(np.floor(split_counter/n_folds))
        ifold=int(np.floor(split_counter-(irep*n_folds)))        
        
        split_counter=split_counter+1
        
        X_train, X_test = data_trn[train_index,:,:], data[test_index,:,:]
        y_train, y_test = conds_id[train_index], conds_id[test_index]
        
        m_trn=np.zeros((n_conds,nchans,ntps))
        m_tst=np.zeros((n_conds,nchans,ntps))
        
        train_dat_cov = np.empty((0,nchans,ntps))
        
        if balanced_train_dat:
            count_min=min(np.bincount(y_train))
            for idx,c in enumerate(u_conds):
                temp_dat=X_train[y_train==c,:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m_trn[idx,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                if balanced_cov:
                    if residual_cov:
                        train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                    else:
                        train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
        else:
            for idx,c in enumerate(u_conds):
                m_trn[idx,:,:]=np.mean(X_train[y_train==c,:,:],axis=0)
                     
        if balanced_test_dat:
            count_min=min(np.bincount(y_test))
            for idx,c in enumerate(u_conds):
                temp_dat=X_test[y_test==c,:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m_tst[idx,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
        else:
            for idx,c in enumerate(u_conds):
                m_tst[idx,:,:]=np.mean(X_test[y_test==c,:,:],axis=0)
                
        if not balanced_cov:
            train_dat_cov=X_train
            
        if cov_metric and not cov_tp:
            train_dat_cov=np.mean(train_dat_cov,axis=-1,keepdims=False) 
            sigma=pinv(covdiag(train_dat_cov))       
        for itp in range(ntps):
            sigma=pinv(covdiag(train_dat_cov[:,:,itp])) 
            
            for icond in range(n_conds):
                temp_dists=np.matmul(np.matmul((m_trn[icond,:,itp]-m_trn[:,:,itp]),sigma),(m_tst[icond,:,itp]-m_tst[:,:,itp]).T)
                RDM[irep,ifold,:,icond,itp]=np.diag(temp_dists)
        bar.next()
    
    bar.finish()            
    RDM=np.mean(RDM,axis=1)
    if average:
        RDM=np.mean(RDM,axis=0)
    
    return RDM,cond_combs
#%%
def RSA_GLM(RDM,models,zscore_models=True,zscore_RDM=True,ddof=0,residual=False):
    
    if len(models.shape)<3:
        models=np.expand_dims(models,axis=-1)
        
    X=np.ones(shape=(RDM.shape[0]*RDM.shape[0],models.shape[-1]+1))
    
    for m in range(models.shape[-1]):
        model_temp=np.squeeze(models[:,:,m]).flatten()
        if zscore_models:           
            X[:,m]=zscore(model_temp,ddof=ddof)
        else:
            X[:,m]=model_temp
            
    if zscore_RDM:
        Y=zscore(RDM.flatten(),ddof=ddof)
    else:
        Y=RDM.flatten()
        
    betas=np.matmul(pinv(X),Y.T)
    
    if residual:
        RDM_res=Y.T-np.matmul(X,betas)
        RDM_res=np.reshape(RDM_res,(RDM.shape[0],RDM.shape[0]))
    else:
        RDM_res=[]
        
    return betas,RDM_res

#%%
from decimal import Decimal, getcontext
getcontext().prec = 100
def RSA_GLM2(RDM,models,zscore_models=True,zscore_RDM=True,residual=False):
    
    if len(models.shape)<3:
        models=np.expand_dims(models,axis=-1)
        
    X=np.ones(shape=(RDM.shape[0]*RDM.shape[0],models.shape[-1]+1))
    
    for m in range(models.shape[-1]):
        model_temp=np.squeeze(models[:,:,m]).flatten()
        if zscore_models:           
            X[:,m]=zscore(model_temp)
        else:
            X[:,m]=model_temp
            
    if zscore_RDM:
        Y=zscore(RDM.flatten())
    else:
        Y=RDM.flatten()
        
    betas=np.matmul(pinv(X),Y.T)
    
    if residual:
        RDM_res=Decimal(Y.T)-Decimal(np.matmul(X,betas))
        RDM_res=np.reshape(RDM_res,(RDM.shape[0],RDM.shape[0]))
    else:
        RDM_res=[]
        
    return betas,RDM_res

#%%
def mahal_CV_RSA_ct(data,conditions,n_folds=8,n_reps=100,data_trn=None,cov_metric='covdiag',cov_tp=True,balanced_train_dat=True,balanced_test_dat=True,balanced_cov=True,residual_cov=False,null_decoding=False,average=True):
    
    if data_trn is None:
        data_trn=data
           
    ntrls, nchans, ntps=np.shape(data)  
    
    # get all unique conditions combinations
    cond_combs= np.unique(conditions, axis=0)  
    _, conds_id= np.unique(np.concatenate((conditions,cond_combs)),axis=0,return_inverse=True)
    conds_id=conds_id[:ntrls]
    u_conds=np.unique(conds_id)
    n_conds=len(u_conds)
    
    #%%
    RDM=np.zeros((n_reps,n_conds,n_conds,ntps,ntps))
    RDM_folds=np.zeros((n_folds,n_conds,n_conds,ntps,ntps))
    RDM[:]=np.nan
    RDM_folds[:]=np.nan
    
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps)
    
    x_dummy=np.zeros(ntrls)
    
    split_counter=0
    
    # bar = ChargingBar('Processing', max=ntps*n_reps*n_folds*n_conds)
    bar = ChargingBar('Processing', max=n_reps*n_folds)
    
    for train_index, test_index in rskf.split(X=x_dummy,y=conds_id):
        
        irep=int(np.floor(split_counter/n_folds))
        ifold=int(np.floor(split_counter-(irep*n_folds)))        
        
        split_counter=split_counter+1
        
        X_train, X_test = data_trn[train_index,:,:], data[test_index,:,:]
        y_train, y_test = conds_id[train_index], conds_id[test_index]
        
        m_trn=np.zeros((n_conds,nchans,ntps))
        m_tst=np.zeros((n_conds,nchans,ntps))
        
        train_dat_cov = np.empty((0,nchans,ntps))
        
        if balanced_train_dat:
            count_min=min(np.bincount(y_train))
            for idx,c in enumerate(u_conds):
                temp_dat=X_train[y_train==c,:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m_trn[idx,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
                if balanced_cov:
                    if residual_cov:
                        train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:]-np.mean(temp_dat[ind,:,:],axis=0), axis=0)
                    else:
                        train_dat_cov = np.append(train_dat_cov, temp_dat[ind,:,:], axis=0)
        else:
            for idx,c in enumerate(u_conds):
                m_trn[idx,:,:]=np.mean(X_train[y_train==c,:,:],axis=0)
                     
        if balanced_test_dat:
            count_min=min(np.bincount(y_test))
            for idx,c in enumerate(u_conds):
                temp_dat=X_test[y_test==c,:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m_tst[idx,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
        else:
            for idx,c in enumerate(u_conds):
                m_tst[idx,:,:]=np.mean(X_test[y_test==c,:,:],axis=0)
                
        if not balanced_cov:
            train_dat_cov=X_train
            
        if cov_metric and not cov_tp:
            train_dat_cov=np.mean(train_dat_cov,axis=-1,keepdims=False) 
            sigma=pinv(covdiag(train_dat_cov))       
        for itp in range(ntps):
            sigma=pinv(covdiag(train_dat_cov[:,:,itp]))
            for itp2 in range(ntps):
                for icond in range(n_conds):
                    temp_dists=np.matmul(np.matmul((m_trn[icond,:,itp]-m_trn[:,:,itp]),sigma),(m_tst[icond,:,itp2]-m_tst[:,:,itp2]).T)
                    RDM_folds[ifold,:,icond,itp,itp2]=np.diag(temp_dists)             
        if ifold+1==n_folds:
            RDM[irep,:,:,:,:]=np.mean(RDM_folds,axis=0)
            
        bar.next()
    #%%
    bar.finish()            
    if average:
        RDM=np.mean(RDM,axis=0)
    
    return RDM,cond_combs

#%%
def euclid_CV_RSA(data,conditions,n_folds=8,n_reps=100,data_trn=None,balanced_train_dat=True,balanced_test_dat=True,null_decoding=False,average=True):
    
    if len(data.shape)<3:
        data=np.expand_dims(data,axis=-1)
    if data_trn is None:
        data_trn=data
    
    if len(data_trn.shape)<3:
        data_trn=np.expand_dims(data_trn,axis=-1)
    
        
    ntrls, nchans, ntps=np.shape(data)  
    
    # get all unique conditions combinations
    cond_combs= np.unique(conditions, axis=0)  
    _, conds_id= np.unique(np.concatenate((conditions,cond_combs)),axis=0,return_inverse=True)
    conds_id=conds_id[:ntrls]
    u_conds=np.unique(conds_id)
    n_conds=len(u_conds)
    
    #%%
    RDM=np.zeros((n_reps,n_folds,n_conds,n_conds,ntps))
    RDM[:]=np.nan
    
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps)
    
    x_dummy=np.zeros(ntrls)
    
    split_counter=0
    
    # bar = ChargingBar('Processing', max=ntps*n_reps*n_folds*n_conds)
    bar = ChargingBar('Processing', max=n_reps*n_folds)
    
    for train_index, test_index in rskf.split(X=x_dummy,y=conds_id):
        
        irep=int(np.floor(split_counter/n_folds))
        ifold=int(np.floor(split_counter-(irep*n_folds)))        
        
        split_counter=split_counter+1
        
        X_train, X_test = data_trn[train_index,:,:], data[test_index,:,:]
        y_train, y_test = conds_id[train_index], conds_id[test_index]
        
        m_trn=np.zeros((n_conds,nchans,ntps))
        m_tst=np.zeros((n_conds,nchans,ntps))
        
        if balanced_train_dat:
            count_min=min(np.bincount(y_train))
            for idx,c in enumerate(u_conds):
                temp_dat=X_train[y_train==c,:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m_trn[idx,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
        else:
            for idx,c in enumerate(u_conds):
                m_trn[idx,:,:]=np.mean(X_train[y_train==c,:,:],axis=0)
                     
        if balanced_test_dat:
            count_min=min(np.bincount(y_test))
            for idx,c in enumerate(u_conds):
                temp_dat=X_test[y_test==c,:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m_tst[idx,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
        else:
            for idx,c in enumerate(u_conds):
                m_tst[idx,:,:]=np.mean(X_test[y_test==c,:,:],axis=0)
                
                  
        for itp in range(ntps):
            for icond in range(n_conds):
                temp_dists=np.matmul((m_trn[icond,:,itp]-m_trn[:,:,itp]),(m_tst[icond,:,itp]-m_tst[:,:,itp]).T)
                RDM[irep,ifold,:,icond,itp]=np.diag(temp_dists)
        bar.next()
    
    bar.finish()            
    RDM=np.mean(RDM,axis=1)
    if average:
        RDM=np.mean(RDM,axis=0)
    
    return RDM,cond_combs

#%% don't use
def corr_spear_CV_RSA(data,conditions,n_folds=8,n_reps=100,data_trn=None,balanced_train_dat=True,balanced_test_dat=True,null_decoding=False,average=True):
    
    if len(data.shape)<3:
        data=np.expand_dims(data,axis=-1)
    if data_trn is None:
        data_trn=data
    
    if len(data_trn.shape)<3:
        data_trn=np.expand_dims(data_trn,axis=-1)
    
        
    ntrls, nchans, ntps=np.shape(data)  
    
    # get all unique conditions combinations
    cond_combs= np.unique(conditions, axis=0)  
    _, conds_id= np.unique(np.concatenate((conditions,cond_combs)),axis=0,return_inverse=True)
    conds_id=conds_id[:ntrls]
    u_conds=np.unique(conds_id)
    n_conds=len(u_conds)
    
    #%%
    RDM=np.zeros((n_reps,n_folds,n_conds,n_conds,ntps))
    RDM[:]=np.nan
    
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps)
    
    x_dummy=np.zeros(ntrls)
    
    split_counter=0
    
    # bar = ChargingBar('Processing', max=ntps*n_reps*n_folds*n_conds)
    bar = ChargingBar('Processing', max=n_reps*n_folds)
    
    for train_index, test_index in rskf.split(X=x_dummy,y=conds_id):
        
        irep=int(np.floor(split_counter/n_folds))
        ifold=int(np.floor(split_counter-(irep*n_folds)))        
        
        split_counter=split_counter+1
        
        X_train, X_test = data_trn[train_index,:,:], data[test_index,:,:]
        y_train, y_test = conds_id[train_index], conds_id[test_index]
        
        m_trn=np.zeros((n_conds,nchans,ntps))
        m_tst=np.zeros((n_conds,nchans,ntps))
        
        if balanced_train_dat:
            count_min=min(np.bincount(y_train))
            for idx,c in enumerate(u_conds):
                temp_dat=X_train[y_train==c,:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m_trn[idx,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
        else:
            for idx,c in enumerate(u_conds):
                m_trn[idx,:,:]=np.mean(X_train[y_train==c,:,:],axis=0)
                     
        if balanced_test_dat:
            count_min=min(np.bincount(y_test))
            for idx,c in enumerate(u_conds):
                temp_dat=X_test[y_test==c,:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m_tst[idx,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
        else:
            for idx,c in enumerate(u_conds):
                m_tst[idx,:,:]=np.mean(X_test[y_test==c,:,:],axis=0)
                
                  
        for itp in range(ntps):
            for icond in range(n_conds):
                for icond2 in range(n_conds):
                    # out=pearsonr(m_trn[icond,:,itp],m_tst[icond2,:,itp])
                    out=spearmanr(m_trn[icond,:,itp],m_tst[icond2,:,itp])
                    
                    RDM[irep,ifold,icond2,icond,itp]=out[0]
                
                # m_trn_df=pd.DataFrame(m_trn[:,:,itp].T)
                
                # corr_trn=-(np.asarray(m_trn_df.corr())-1)
                
                # m_tst_df=pd.DataFrame(m_tst[:,:,itp].T)
                
                # corr_tst=-(np.asarray(m_tst_df.corr())-1)
                
                # corrs_diff=corr_trn-corr_tst
                
                # m_trn_df.corrwith(m_tst_df,method='pearson')
                
                # # m_trn_df[:].corr(m_tst_df)
                
                # corr_tst=-(np.asarray(m_tst_df.corr())-1)
                
    
                
                # corr_trn_tst=np.corrcoef(m_trn[:,:,itp],m_tst[:,:,itp],'full')
                
                # dist_trn=(m_trn[icond,:,itp]-m_trn[:,:,itp])
                # dist_tst=(m_tst[icond,:,itp]-m_tst[:,:,itp])
                
                # pearsonr(m_trn[:,:,itp],m_tst[:,:,itp])
                
                # temp1=np.corrcoef(m_trn[icond,:,itp],m_tst[:,:,itp])
                # temp1=temp1[~np.eye(temp1.shape[0],dtype=bool)].reshape(temp1.shape[0],-1)
                
                
                # temp_dists=np.matmul((m_trn[icond,:,itp]-m_trn[:,:,itp]),(m_tst[icond,:,itp]-m_tst[:,:,itp]).T)
                # RDM[irep,ifold,:,icond,itp]=np.diag(temp_dists)
        bar.next()
    
    bar.finish()            
    RDM=np.mean(RDM,axis=1)
    if average:
        RDM=np.mean(RDM,axis=0)
    
    return RDM,cond_combs

#%% don't use
def corr_pears_CV_RSA(data,conditions,n_folds=8,n_reps=100,data_trn=None,balanced_train_dat=True,balanced_test_dat=True,null_decoding=False,average=True):
    
    if len(data.shape)<3:
        data=np.expand_dims(data,axis=-1)
    if data_trn is None:
        data_trn=data
    
    if len(data_trn.shape)<3:
        data_trn=np.expand_dims(data_trn,axis=-1)
    
        
    ntrls, nchans, ntps=np.shape(data)  
    
    # get all unique conditions combinations
    cond_combs= np.unique(conditions, axis=0)  
    _, conds_id= np.unique(np.concatenate((conditions,cond_combs)),axis=0,return_inverse=True)
    conds_id=conds_id[:ntrls]
    u_conds=np.unique(conds_id)
    n_conds=len(u_conds)
    
    #%%
    RDM=np.zeros((n_reps,n_folds,n_conds,n_conds,ntps))
    RDM[:]=np.nan
    
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_reps)
    
    x_dummy=np.zeros(ntrls)
    
    split_counter=0
    
    # bar = ChargingBar('Processing', max=ntps*n_reps*n_folds*n_conds)
    bar = ChargingBar('Processing', max=n_reps*n_folds)
    
    for train_index, test_index in rskf.split(X=x_dummy,y=conds_id):
        
        irep=int(np.floor(split_counter/n_folds))
        ifold=int(np.floor(split_counter-(irep*n_folds)))        
        
        split_counter=split_counter+1
        
        X_train, X_test = data_trn[train_index,:,:], data[test_index,:,:]
        y_train, y_test = conds_id[train_index], conds_id[test_index]
        
        m_trn=np.zeros((n_conds,nchans,ntps))
        m_tst=np.zeros((n_conds,nchans,ntps))
        
        if balanced_train_dat:
            count_min=min(np.bincount(y_train))
            for idx,c in enumerate(u_conds):
                temp_dat=X_train[y_train==c,:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m_trn[idx,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
        else:
            for idx,c in enumerate(u_conds):
                m_trn[idx,:,:]=np.mean(X_train[y_train==c,:,:],axis=0)
                     
        if balanced_test_dat:
            count_min=min(np.bincount(y_test))
            for idx,c in enumerate(u_conds):
                temp_dat=X_test[y_test==c,:,:]
                ind=random.sample(list(range(temp_dat.shape[0])),count_min)
                m_tst[idx,:,:]=np.mean(temp_dat[ind,:,:],axis=0)
        else:
            for idx,c in enumerate(u_conds):
                m_tst[idx,:,:]=np.mean(X_test[y_test==c,:,:],axis=0)
                
                  
        for itp in range(ntps):
            for icond in range(n_conds):
                for icond2 in range(n_conds):
                    out=pearsonr(m_trn[icond,:,itp],m_tst[icond2,:,itp])
                    # out=spearmanr(m_trn[icond,:,itp],m_tst[icond2,:,itp])
                    
                    RDM[irep,ifold,icond2,icond,itp]=out[0]
                
                # m_trn_df=pd.DataFrame(m_trn[:,:,itp].T)
                
                # corr_trn=-(np.asarray(m_trn_df.corr())-1)
                
                # m_tst_df=pd.DataFrame(m_tst[:,:,itp].T)
                
                # corr_tst=-(np.asarray(m_tst_df.corr())-1)
                
                # corrs_diff=corr_trn-corr_tst
                
                # m_trn_df.corrwith(m_tst_df,method='pearson')
                
                # # m_trn_df[:].corr(m_tst_df)
                
                # corr_tst=-(np.asarray(m_tst_df.corr())-1)
                
    
                
                # corr_trn_tst=np.corrcoef(m_trn[:,:,itp],m_tst[:,:,itp],'full')
                
                # dist_trn=(m_trn[icond,:,itp]-m_trn[:,:,itp])
                # dist_tst=(m_tst[icond,:,itp]-m_tst[:,:,itp])
                
                # pearsonr(m_trn[:,:,itp],m_tst[:,:,itp])
                
                # temp1=np.corrcoef(m_trn[icond,:,itp],m_tst[:,:,itp])
                # temp1=temp1[~np.eye(temp1.shape[0],dtype=bool)].reshape(temp1.shape[0],-1)
                
                
                # temp_dists=np.matmul((m_trn[icond,:,itp]-m_trn[:,:,itp]),(m_tst[icond,:,itp]-m_tst[:,:,itp]).T)
                # RDM[irep,ifold,:,icond,itp]=np.diag(temp_dists)
        bar.next()
    
    bar.finish()            
    RDM=np.mean(RDM,axis=1)
    if average:
        RDM=np.mean(RDM,axis=0)
    
    return RDM,cond_combs
            
        
    
    

    
                
            
            
                
            
        
        
              
        
    
    

    
                
            
            
                
            
        
        
    