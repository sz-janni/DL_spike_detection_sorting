import h5py
import numpy as np
import csv
from matplotlib import pyplot as plt

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

filename = "./data/synth_1152_bme.h5"
filename_labels="./data/synth_1152_bme.spt"
Fs=20000
length_in_seconds=50
dt=length_in_seconds/Fs
all_labels=None

#Load all labels
with open(filename_labels, 'r') as f:
    all_labels = tuple(csv.reader(f,delimiter=' '))
tot_channel_num=get_tot_channel_num(filename)
data=None

#Load electrode array channels one by one
for i in range(0,tot_channel_num):
    data=adc_to_mv(load_channel(filename,i))
    data_t=np.array(list(range(data.shape[0])))
    data_t=data_t/Fs
    labels=[]
    labels_t=[]
    for row in all_labels:
        row=list(filter(lambda x:x!='', row))  
        try:
            if int(row[3])==i+1:
                labels.append(int(row[4]))
                labels_t.append(float(row[0]))
        except:
            pass
    labels=np.array(labels)
    labels_t=np.array(labels_t)
    ones=np.ones(len(labels_t))+30
    plt.plot(data_t,data)
    labels_in_channel=np.unique(labels_t)
    for l in labels_in_channel:
        plt.scatter(labels_t[labels_t==l],data[data_t==labels_t[labels_t==l]])
    plt.show()
            
...
