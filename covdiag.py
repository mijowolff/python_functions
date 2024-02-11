
import numpy as np

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
