import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging
def opt_cnn(x_train,y_train,x_val,y_val):
    def objective(trial):
        #batch_size=round(len(x_train)/10)
        batch_size=trial.suggest_categorical('batch_size',[128,256])
        input_shape=(batch_size,1,40)
        #x = keras.backend.random_normal(input_shape)
        model = keras.Sequential()
        model.add(layers.Conv1D(filters=trial.suggest_int('filters', 3, 15),
                      padding='same',
                      kernel_size=trial.suggest_categorical('kernel_size', [3,4, 5]),
                      kernel_regularizer=trial.suggest_categorical('kernel_regularizer', [None,'l1', 'l1_l2']),
                      bias_regularizer=trial.suggest_categorical('bias_regularizer', [None,'l1', 'l1_l2']),
                      activation=trial.suggest_categorical('activation', ['swish','sigmoid', 'relu']), 
                      input_shape=input_shape[1:],name="conv"))
        model.add(layers.Flatten()
        )
        model.add(layers.Dense(5,
                    activation=trial.suggest_categorical('activation_dense', ['swish','sigmoid', 'relu']),
                    kernel_regularizer=trial.suggest_categorical('kernel_regularizer_dense1', [None,'l1', 'l1_l2']),
                    bias_regularizer=trial.suggest_categorical('bias_regularizer_dense1', [None,'l1', 'l1_l2']),name="out1")
        )
        model.add(layers.Dense(5,
                    activation='softmax',
                    kernel_regularizer=trial.suggest_categorical('kernel_regularizer_dense2', [None,'l1', 'l1_l2']),
                    bias_regularizer=trial.suggest_categorical('bias_regularizer_dense2', [None,'l1', 'l1_l2']),name="out")
        )
        model.compile(
        loss="categorical_crossentropy",  metrics=["accuracy","AUC"] , optimizer=trial.suggest_categorical('optimizer',['adam','nadam'])
        )
        callback = keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5,min_delta=0.001)
        model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=500,
        callbacks=[callback],
        validation_data=(x_val, y_val),
        verbose=False,
        )

    # Evaluate the model accuracy on the validation set.
        score = model.evaluate(x_val, y_val, verbose=0)
        score=(score[1]+score[2])/2
        return score

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler("./results/optuna_logs/Optuna.log", mode="w"))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler() 
    study = optuna.create_study(direction='maximize',sampler = optuna.samplers.TPESampler(seed=10,multivariate=True,n_startup_trials=10)) #TPESampler
    study.optimize(objective, n_trials=100)
    plot_par_cor=optuna.visualization.plot_parallel_coordinate(study)
    plot_import=optuna.visualization.plot_param_importances(study)
    plot_history=optuna.visualization.plot_optimization_history(study)
    plot_par_cor.write_html("./results/optuna_logs/par_cor_cv_.html")
    plot_import.write_html("./results//optuna_logs/importance_cv_.html")
    plot_history.write_html("./results//optuna_logs/history_cv_.html")
    return study.best_params
def opt_autoencoder():
    def objective(trial):
        pass
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=250)