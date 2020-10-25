import h5py
import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy import signal
import multiprocessing as mp
from detect_spikes import detect_spikes

def extract_spike_waveforms(data,spike_labels):
    all_waveforms=np.empty((len(spike_labels),40))
    for idx,i in enumerate(spike_labels):
        waveform=np.array(data[i-20:i+20],dtype=np.float32)
        if len(waveform)<40:
            zeropadding=np.zeros((40-len(waveform)))
            waveform=np.hstack((waveform,zeropadding))
        all_waveforms[idx]=waveform
    return all_waveforms
    
    
def load_labels(fname):
    with open(fname, 'r') as f:
        all_labels = tuple(csv.reader(f,delimiter=' '))
    return all_labels
def move_label_to_peak(data,labels_t,data_t):
    max_length_to_check_peak=50
    gradient=np.gradient(data)
    all_peaks=[]
    for i in labels_t:
        idx=int(np.argwhere(i==data_t))
        
        #Gradient based peak moving
        # gradient_pos=bool(gradient[idx]>0)
        # if gradient_pos:
        #     found_peaks=signal.find_peaks(data[idx:idx+max_length_to_check_peak])
        # else:
        #     found_peaks=signal.find_peaks(-1*data[idx:idx+max_length_to_check_peak])
        # if len(found_peaks)<1:
        #     print('No peak found')
        # else:
        #     found_peak=found_peaks[0][0]
        #     all_peaks.append(found_peak+idx)
        
        #Higher peak based peak moving
        gradient_pos=bool(gradient[idx]>0)
        if gradient_pos:
            post_found_peak=signal.find_peaks(data[idx-2:idx+max_length_to_check_peak])
            pre_found_peak=signal.find_peaks(-1*data[idx-max_length_to_check_peak:idx+2])
            try:
                post_found_peak=post_found_peak[0][0]
            except:
                try:
                    pre_found_peak=pre_found_peak[0][-1]
                    all_peaks.append((idx-max_length_to_check_peak)+pre_found_peak)
                    continue
                except:
                    print('No peak found')
                    continue
            try:
                pre_found_peak=pre_found_peak[0][-1]
            except:
                all_peaks.append(post_found_peak+(idx-2))
                continue

            
            if abs(data[post_found_peak+(idx-2)])>abs(data[(idx-max_length_to_check_peak)+pre_found_peak]):
                all_peaks.append(post_found_peak+(idx-2))
            else:
                all_peaks.append((idx-max_length_to_check_peak)+pre_found_peak)
        else:
            post_found_peak=signal.find_peaks(-1*data[idx-2:idx+max_length_to_check_peak])
            pre_found_peak=signal.find_peaks(data[idx-max_length_to_check_peak:idx+2])
            try:
                post_found_peak=post_found_peak[0][0]
            except:
                try:
                    pre_found_peak=pre_found_peak[0][-1]
                    all_peaks.append((idx-max_length_to_check_peak)+pre_found_peak)
                    continue
                except:
                    print('No peak found')
                    continue
            try:
                pre_found_peak=pre_found_peak[0][-1]
            except:
                all_peaks.append(post_found_peak+(idx-2))
                continue
                
                
            if abs(data[post_found_peak+(idx-2)])>abs(data[(idx-max_length_to_check_peak)+pre_found_peak]):
                all_peaks.append(post_found_peak+(idx-2))
            else:
                all_peaks.append((idx-max_length_to_check_peak)+pre_found_peak)

    return all_peaks
                

def get_tot_channel_num(filename):
    with h5py.File(filename, "r") as f:
        data = f.get('signals')[0,:]
        data=np.array(data).shape[0]
    return data
def load_channel(filename,channel_idx):
    with h5py.File(filename, "r") as f:
        data = f.get('signals')[:,channel_idx]
        data=np.array(data)
    return data
def adc_to_mv(adc):
    return 0.194708 * (adc - 32768.0)
#Load electrode array channels one by one
def process_electrode_channels(start,end,threadnum,spikes):
    ap_length=40
    all_aps=len(spikes['all_labels'])
    all_waveforms=np.empty((all_aps,ap_length),dtype=np.float32)
    all_labels=[]
    counter=0
    for i in range(start,end):
        if (i-start)%25==0:
            print(str(round((i-start)/(end-start)*100,1))+ '% of channels done for thread '+ str(threadnum))
        data=adc_to_mv(load_channel(spikes['filename'],i))
        b,a=signal.butter(2,(150,2500),btype='bandpass',fs=spikes['Fs'])
        data=signal.filtfilt(b,a,data)
        data_t=np.array(list(range(data.shape[0])))
        data_t=data_t/spikes['Fs']
        labels=[]
        labels_t=[]
        idxs_to_delete=[]
        for rowidx,row in enumerate(spikes['all_labels']):
            row=list(filter(lambda x:x!='', row))  
            try:
                if int(row[3])==i+1:
                    labels.append(int(row[4]))
                    labels_t.append(float(row[0]))
                    idxs_to_delete.append(rowidx)
                    
            except:
                pass
        for k,idx in enumerate(idxs_to_delete):
            #Deletion of earlier elements shifts idxs
            spikes['all_labels'].pop(idx-k)
                    
                
        labels=np.array(labels)
        labels_t=np.array(labels_t)
        labels_t=move_label_to_peak(data,labels_t,data_t)
        #found_peaks=detect_spikes(data,spikes['Fs'])
        waveforms=extract_spike_waveforms(data,labels_t)
        num_ap_waveforms=waveforms.shape[0]
        all_waveforms[counter:counter+num_ap_waveforms]=waveforms
        all_labels.append(labels)
        counter+=num_ap_waveforms
    return all_waveforms[~np.all(all_waveforms == 0, axis=1)],all_labels
        # plt.plot(data_t,data)
        # labels_in_channel=np.unique(labels_t)
        # for l in labels_in_channel:
        #     plt.scatter(labels_t[labels_t==l],data[data_t==labels_t[labels_t==l]])
        # plt.show()

         
...
