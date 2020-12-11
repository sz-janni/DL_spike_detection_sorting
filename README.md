Team name (csapatnév): TaRéJ

Members (csapattagok):
- János Szalma
- Tamás Nagy
- ~~Réka Böröcz~~ (left)

# DL_spike_detection_sorting
Spike detection and sorting using deep learning


# Usage

Download data and labels from:\
https://drive.google.com/file/d/1nkpakzO_nnagPe9n4FDsBOYkRBOdb7M6/view?usp=sharing \
and\
https://drive.google.com/file/d/1D36Y5mtn6yx4nS_54DVUyf_F0Sk3tV09/view?usp=sharing \
Copy files into main folder \
Run train_test_model.py \
To lower runtime decrease setting variables N_TRIALS and N_REPEATS


# Scripts

## Main_preprocess.py
Process spike data with multithread option

## Process_spike_data.py
Functions to load hdf5 and spt files, match labels to data, plot and change label locations

## Detect_spikes.py
Adaptive spike detection algorithm (https://www.hindawi.com/journals/cin/2010/659050/)
(Currently not used adaptively)

## xmeans.ipynb
Testing the AE based denoising (denoising.py) and using the X-means algorithm for clustering.

## Train_test_model.py
Train and test deep learning model to classify spike waveforms

## Hyperparam_opt.py
Optimize neural network hyperparameters using Optuna and TPE sampling

## Visualisation.ipynb
Extract waveform features, calculate basis statistics, PCA and visualize




