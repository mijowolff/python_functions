# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:42:33 2021

@author: Michael

combine channel and time dimensions, for potentially improved subsequent multivariate decoding

"""
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
import warnings

def dat_prep_4d_time_course(data,time_dat,toi,window_length=100,span=10,steps=10,relative_baseline=True,window_center='right',in_ms=True):
    
    """
    create sliding window that combines channels and surrounding time-points
    
    returns:    dat_new (formatted data, with combined channel and time dimensions)
                time_new (time-points associated with the time dimension of dat_new)
    
    data        = trial by channel by time
    time_dat    = time points associated with data
    toi         = time of interest of the time-course
        make sure that the time dimension of the input data is long enough to include the edges of the sliding window of toi;
        (depends on toi, window_length, and window_center)
        if it isn't, the outer edge(s) of toi that don't fit will be ignored
        
    window_length   = in ms or time-points, length of sliding window that time-points are combined over, should be a multiple of span (see below)
    span            = in ms or time-points, length of each segment that time-points are averaged over in the sliding window, 
                        used for downsampling the time-dimension before combining it with the channel dimension
    steps           = in ms or time-points, the steps the sliding window is moving in (to downsample time-dimension of output data)
    
    relative_baseline = True/False (default True), whether or not the relative baseline is taken within each time-window, which 
                        removes the average signal (solves the baseline issue in EEG data, but leaves only dynamics, i.e. signal 
                        that changes within the length of the sliding window)
    window_center     = left/center/right (default right), the center of the sliding window, 
                        'right' is chosen as default as it allows for interpration of effect onsets (but not offsets!)
                        since only previous time-points are included in the sliding window
    in_ms             = True/False (default True), whether magnitutes are in ms or time-points
    
    """
    
    
    time_dat=np.squeeze(time_dat)
    hz=float(np.round(1/np.diff(time_dat[:2]))) # determine sample-rate of input data
    
    if in_ms:
        # convert ms inputs to time-points using sample-rate of input data    
        window_length=int(window_length/(1000/hz))
        span=int(span/(1000/hz))
        steps=int(steps/(1000/hz)) 
    
    
    # just to ensure all desired time-points are included...
    toi_new=np.empty(2)
    toi_new[0]=toi[0]-(1000/hz)/4000
    toi_new[1]=toi[1]+(1000/hz)/4000
    
    # issue warning and convert span/steps to minimum if input is smaller than input data sample hz
    if span<1:
        warnings.warn(' "span" is smaller than input sample hz, using minimum instead!')
        span=int(1)        
    if steps<1:
        warnings.warn(' "steps" is smaller than input sample hz, using minimum instead!')
        steps=int(1)
    
    # determine left and right window boundaries
    if window_center=='right':        
        wl=-(window_length-1)
        wr=0
    elif window_center=='center':
        wl=int(-window_length/2)
        wr=int(window_length/2)
    elif window_center=='left':
        wl=0
        wr=window_length-1
        
    # check if sliding window and toi fit into input data, 
    # if not, issue warning and trim toi
    ind_left=np.argmin(abs(time_dat-toi_new[0]))+wl
    ind_right=np.argmin(abs(time_dat-toi_new[1]))+wr
    if ind_left<0:
        toi_new[0]=time_dat[-wl] 
        toi_new[0]=toi_new[0]-(1000/hz)/4000
    if ind_right>len(time_dat):
        toi_new[1]=time_dat[-1-wr] 
        toi_new[1]=toi_new[1]+(1000/hz)/4000
    if ind_left<0 or ind_right>len(time_dat):
        warnings.warn(' the chosen "toi" and "window_length" do not fit, "toi" is therefore trimmed')
        
    
    n_trls, n_chans,n_tps_temp=np.shape(data) 
    
    # output time          
    time_new=time_dat[(time_dat>toi_new[0])&(time_dat<toi_new[1])]
    time_new=time_new[0::int(steps)]
    
    n_tps=len(time_new)    
                
    dat_new=np.empty([n_trls,int((window_length/span)*n_chans),n_tps])
    
    for tp in range(n_tps): # loop over each time-point to make 
        ind=np.argmin(abs(time_dat-time_new[tp]))
        inds=list(range(ind+wl,ind+wr+1))
        
        if relative_baseline:
            # subtract mean signal
            w_mean=np.mean(data[:,:,inds],axis=-1,keepdims=True)
            temp_dat=data[:,:,inds]-np.tile(w_mean,(1,1,len(inds)))
        else:
            temp_dat=data[:,:,inds]
        
        # downsample data in time-window
        temp_dat_s=uniform_filter1d(temp_dat,span,axis=-1,mode='reflect')              
        temp_dat_s_ds=np.squeeze(temp_dat_s[:,:,int(np.floor(span/2))::span])  
        
        # combine channel and time dimensions                                     
        dat_new[:,:,tp] = temp_dat_s_ds.reshape(temp_dat_s_ds.shape[0],temp_dat_s_ds.shape[1]*temp_dat_s_ds.shape[2]) 
    
    return dat_new,time_new




def dat_prep_4d_section(data,time_dat=None,toi=None,span=10,hz=500,relative_baseline=True,in_ms=True):    
    
    if toi is not None:
        if time_dat is not None:
            hz=float(np.round(1/np.diff(time_dat[:2]))) # determine sample-rate of input data
            time_dat=np.squeeze(time_dat)
            toi_new=np.zeros(2)
            toi_new[1]=toi[1]+(1000/hz)/4000 # just in case, to avoid possibly missing a time-point...
            toi_new[0]=toi[0]
            data=data[:,:,(time_dat>toi_new[0])&(time_dat<toi_new[1])]           
    
    if relative_baseline:
        w_mean=np.mean(data,axis=-1,keepdims=True)
        data=data-np.tile(w_mean,(1,1,data.shape[-1])) 
    
    if time_dat is not None:
        hz=float(np.round(1/np.diff(time_dat[:2])))
    
    if in_ms:         
        span=int(span/(1000/hz))
    
    data2=uniform_filter1d(data,span,axis=-1,mode='constant')
    
    data3=data2[:,:,int(np.floor(span/2))::span] 

    dat_dec=data3.reshape(data3.shape[0],data3.shape[1]*data3.shape[2],order='F')


    return dat_dec