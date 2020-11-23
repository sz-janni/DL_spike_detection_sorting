import tensorflow as tf
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from hyperparam_opt import opt_cnn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import time
from datetime import timedelta,datetime

def logprint(text,file,end='\n'):
    print(text,file=file,flush=True,end=end)
    print(text,end=end)

def stratified_train_test_val_split(data,labels,train=0.05,test=0.9,val=0.05,random_state=0):
    
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train,test_size=test+val, random_state=random_state)
    data=np.asarray(data)
    labels=np.asarray(labels)
    train_index=None
    for train_index, _ in sss.split(data, labels): trainX,trainY=data[train_index],labels[train_index]
    scalevar=1/(test+val)
    num_of_train_toremove=1-(train+test+val)
    test=test*scalevar
    val=val*scalevar
    
    data=np.delete(data, train_index, axis=0)
    labels=np.delete(labels, train_index, axis=0)
    for _ in range(round(num_of_train_toremove)):
        data=np.delete(data,-1,axis=0)
        labels=np.delete(labels,-1,axis=0)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=test,test_size=val, random_state=random_state)
    test_index,val_idx=None, None
    for test_index, val_idx in sss.split(data, labels): 
        testX, testY=data[test_index],labels[test_index]
        valX, valY=data[val_idx],labels[val_idx]
    return trainX,trainY,valX,valY,testX,testY

## SETTINGS
#Number of hyperparameter optimization trials
N_TRIALS=2
#Number of repetitions for training and testing the optimized model, 1 equals no repeats
N_REPEATS=1
#Percentage of data to use for training, use max 0.1, as test is always 0.85 and validation is always 0.05
TRAIN_SIZE=0.05

data_path='./spike_waveform_data.pickle'
label_path='./spike_waveform_labels.pickle'
start_time=time.time()
#sys.stdout = 
dt=datetime.now()
logfile=open('./results/logging_'+str(dt)+'.log', 'w')
logprint('Running script...',logfile)
logprint('Hyperparameter optimization trials: '+str(N_TRIALS),logfile)
logprint('Number of repeats to train test final model: '+str(N_REPEATS),logfile)
logprint('Training data percentage: '+str(TRAIN_SIZE),logfile)

# Load, sort and preprocess data
data = pd.read_pickle(data_path)
labels = pd.read_pickle(label_path)
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
x_train,y_train,x_val,y_val,x_test,y_test=stratified_train_test_val_split(data,labels,train=TRAIN_SIZE)
scaler = MinMaxScaler()
scaler=scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_val=scaler.transform(x_val)
x_test=scaler.transform(x_test)
x_train=np.reshape(x_train,(-1,1,40))
x_val=np.reshape(x_val,(-1,1,40))
x_test=np.reshape(x_test,(-1,1,40))

#Optimize hyperparameters
best_params=opt_cnn(x_train,y_train,x_val,y_val,N_TRIALS,dt)
#Train and test optimized model with possible reruns
for _ in range(N_REPEATS):
    batch_size=best_params['batch_size']
    input_shape=(batch_size,1,40)
    model = keras.Sequential()
    model.add(layers.Conv1D(input_shape=input_shape[1:],padding='same',
                            filters=best_params['filters'],kernel_size=best_params['kernel_size'],kernel_regularizer=best_params['kernel_regularizer'],
                            bias_regularizer=best_params['bias_regularizer'],activation=best_params['activation']))
    model.add(layers.Flatten())
    model.add(layers.Dense(5,
                        activation=best_params['activation_dense'],
                        kernel_regularizer=best_params['kernel_regularizer_dense1'],
                        bias_regularizer=best_params['bias_regularizer_dense1']
            ))
    model.add(layers.Dense(5,
                        activation='softmax',kernel_regularizer=best_params['kernel_regularizer_dense2'],bias_regularizer=best_params['bias_regularizer_dense2']))
    model.compile(loss="categorical_crossentropy",  metrics=["accuracy","AUC"] , optimizer='adam')
    callback = keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=15)
    model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=1500,
            callbacks=[callback],
            validation_data=(x_val, y_val),
            verbose=False,
            )

        # Evaluate the model accuracy on the validation set.
    score = model.evaluate(x_test, y_test, verbose=1)

    logprint('Accuracy: '+ str(score[1]),logfile)
    logprint('ROCAUC: '+ str(score[2]),logfile)
end_time=time.time()
elapsed_time=end_time-start_time
logprint('Elapsed time: '+str(timedelta(seconds=elapsed_time)),logfile)
logfile.close()
    #sys.stdout = sys.__stdout__
#print(tf.config.list_physical_devices('GPU'))