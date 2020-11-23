# DL_spike_detection_sorting
Spike detection and sorting using deep learning
## Main_preprocess.py
Process spike data with multithread option
## Process_spike_data.py
Functions to load hdf5 and spt files, match labels to data, plot and change label locations
## Detect_spikes.py
Adaptive spike detection algorithm (https://www.hindawi.com/journals/cin/2010/659050/)
(Currently not used adaptively)
## Train_test_model.py
Train and test deep learning model to classify spike waveforms
## Hyperparam_opt.py
Optimize neural network hyperparameters using Optuna and TPE sampling
## Train_test_val_split_Spike.ipynb
Splits dataset into training, testing, validation.
## Visualisation.ipynb
Extract waveform features, calculate basis statistics, PCA and visualize
# Usage
Run train_test_model.py
To lower runtime decrease setting variables N_TRIALS and N_REPEATS



