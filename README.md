
# 🧠 Neural spike sorting using deep learning-based approaches

<img src="https://www.flaticon.com/svg/static/icons/svg/1756/1756394.svg" align="right"
     alt="Logo" width="120" height="178">

![GitHub issues](https://img.shields.io/github/issues/sz-janni/DL_spike_detection_sorting) ![GitHub commit activity](https://img.shields.io/github/commit-activity/y/sz-janni/DL_spike_detection_sorting) ![Bitbucket open pull requests](https://img.shields.io/bitbucket/pr/sz-janni/DL_spike_detection_sorting) ![GitHub repo size](https://img.shields.io/github/repo-size/sz-janni/DL_spike_detection_sorting)

## Team info
![Bitbucket open pull requests](https://img.shields.io/badge/team%20name-TaR%C3%A9J-blue)

![GitHub contributors](https://img.shields.io/github/contributors/sz-janni/DL_spike_detection_sorting)
- János Szalma
- Tamás Nagy
- ~~Réka Böröcz~~ (left)


## Data and setup

Download data and labels from:\
https://drive.google.com/file/d/1nkpakzO_nnagPe9n4FDsBOYkRBOdb7M6/view?usp=sharing \
and\
https://drive.google.com/file/d/1D36Y5mtn6yx4nS_54DVUyf_F0Sk3tV09/view?usp=sharing \
Copy files into main folder \
Run train_test_model.py \
To lower runtime decrease setting variables N_TRIALS and N_REPEATS

## Final report
The final report can be found in [spike_sorting_final.pdf](https://github.com/sz-janni/DL_spike_detection_sorting/blob/main/spike_sorting_final.pdf).

## Scripts

### main_preprocess.py
Process spike data with multithread option.

### process_spike_data.py
Functions to load hdf5 and spt files, match labels to data, plot and change label locations.

### detect_spikes.py
Adaptive spike detection algorithm (https://www.hindawi.com/journals/cin/2010/659050/)
(Currently not used adaptively).

### visualisation.ipynb
Extract waveform features, calculate basis statistics, PCA and visualize.

### denoising.py
Function library for adding artificial noise to the waveforms, train an Auto Encoder network for noise removal, and use a trained model (from /ae_denoise_model/) to remove noise. \
denoising_test.ipynb shows this in action.

### xmeans.ipynb
Testing the AE based denoising (denoising.py) and using the X-means algorithm for clustering.

### train_test_model.py
Train and test deep learning model to classify spike waveforms.

### hyperparam_opt.py
Optimize neural network hyperparameters using Optuna and TPE sampling.






