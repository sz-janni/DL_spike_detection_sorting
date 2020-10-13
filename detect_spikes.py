import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
def detect_spikes(data,Fs):
    window_size=int((Fs/1000)*10)
    noise_est=None
    window_aggregate_num=100
    window_rms_buffer=[]
    all_noise_estimates=[]
    for i in range(0,len(data),int(window_size)):
        window_rms=np.sqrt(np.mean(data[i:i+window_size]**2))
        window_rms_buffer.append(window_rms)
        if len(window_rms_buffer)==window_aggregate_num:
            if noise_est is None:
                noise_est=np.percentile(window_rms_buffer,25)
            else:
                #Adaptive mode, not currently used
                noise_est=0.8*noise_est+0.2*np.percentile(window_rms_buffer,25)
                all_noise_estimates.append(np.percentile(window_rms_buffer,25))
            window_rms_buffer=[]
    threshold_multiplier=4
    noise_est=np.mean(all_noise_estimates)
    threshold=noise_est*threshold_multiplier
    pos_peaks,pos_peak_heights=signal.find_peaks(data,height=threshold)
    neg_peaks,neg_peak_heights=signal.find_peaks(-1*data,height=threshold)
    neg_peak_heights=-1*neg_peak_heights['peak_heights']
    pos_peak_heights=pos_peak_heights['peak_heights']
    peaks=[pos_peak_heights,neg_peak_heights]
    peak_locs=[pos_peaks,neg_peaks]
    peak_locs = [item for sublist in peak_locs for item in sublist]
    peaks = [item for sublist in peaks for item in sublist]
    idxs=sorted(range(len(peak_locs)), reverse=False, key=lambda k: peak_locs[k])
    peak_locs=np.array(peak_locs)[idxs]
    peaks=np.array(peaks)[idxs]
    while min(np.diff(peak_locs))<20:
        for p in range(len(peak_locs)-1):
            try:
                if peak_locs[p+1]-peak_locs[p]<20:
                    #Different polarities
                    if peaks[p+1]*peaks[p]<0:
                        if abs(peaks[p+1])>abs(peaks[p]):
                            #Remove peaks[p]
                            peaks=np.delete(peaks,p)
                            peak_locs=np.delete(peak_locs,p)
                            p+=-1
                        else:
                            #Remove peaks[p+1]
                            peaks=np.delete(peaks,p+1)
                            peak_locs=np.delete(peak_locs,p+1)
                            p+=-1
                    else: #Same polarities
                        peak_amp_diff=abs(peaks[p+1]-peaks[p])
                        if peaks[p+1]>peaks[p] and peak_amp_diff>peaks[p]*1.5:
                            #Remove peaks[p]
                            peaks=np.delete(peaks,p)
                            peak_locs=np.delete(peak_locs,p)
                            p+=-1
                        elif peaks[p]>peaks[p+1] and peak_amp_diff>peaks[p+1]*1.5:
                            #Remove peaks[p+1]
                            peaks=np.delete(peaks,p+1)
                            peak_locs=np.delete(peak_locs,p+1)
                            p+=-1
                        else:
                            #Remove both
                            peaks=np.delete(peaks,p)
                            peak_locs=np.delete(peak_locs,p)
                            peaks=np.delete(peaks,p+1)
                            peak_locs=np.delete(peak_locs,p+1)
                            p+=-2
            except:
                pass
    return peak_locs                     
        
    
    