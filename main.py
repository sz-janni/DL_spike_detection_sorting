import process_spike_data as psd
import h5py
import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy import signal
import multiprocessing as mp

if __name__== '__main__':
    spikes={}
    spikes['filename'] = "./data/synth_1152_bme.h5"
    spikes['filename_labels']="./data/synth_1152_bme.spt"
    spikes['Fs']=20000
    spikes['length_in_seconds']=50
    spikes['dt']=spikes['length_in_seconds']/spikes['Fs']
    all_labels=None

    #Load all labels
    all_labels=psd.load_labels(spikes['filename_labels'])
    spikes['all_labels']=all_labels
        
    tot_channel_num=psd.get_tot_channel_num(spikes['filename'])
    data=None

    NUM_OF_THREADS_TO_START=1
    offset=int(np.floor(tot_channel_num/NUM_OF_THREADS_TO_START))
    print('Starting with '+str(NUM_OF_THREADS_TO_START)+' threads')
    print(str(offset)+ ' channels to process per thread')
    if NUM_OF_THREADS_TO_START==1:
        psd.process_electrode_channels(0,0+offset,1,spikes)
    else:
        pool = mp.Pool(NUM_OF_THREADS_TO_START)

        processes=[]
        t=1
        for p in range(0,tot_channel_num,offset):
            if p!=tot_channel_num-1:
                processes.append(pool.apply_async(psd.process_electrode_channels, (p,p+offset,t,spikes,)))
            else:
                processes.append(pool.apply_async(psd.process_electrode_channels, (p,tot_channel_num,t,spikes,)))
            t+=1
        pool.close()
        for p in processes:
            p.get()
        pool.join()
    ('...Done') 