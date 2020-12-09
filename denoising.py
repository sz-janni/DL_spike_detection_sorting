import numpy as np
import math


from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
import tensorflow


def add_noise(s, noise_factor = 0.1):
    ns = s + noise_factor * np.random.normal(loc=0.0, scale=np.amax(s), size=s.shape)
    return ns 



def reshape_signal(signal, width=8, height=5):
    s = []
    for i in range(0, len(signal)):
        sample = signal[i]
        sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
        sample = sample.reshape(width, height)
        s.append(sample)
    s = np.array(s)
    reshaped_signal = s.reshape((s.shape[0], s.shape[1], s.shape[2], 1))
    return reshaped_signal
    

    
def data_split(signal, split_ratio):
    percentage = math.floor((1 - split_ratio) * len(signal))
    train, test = signal[:percentage], signal[percentage:]
    return train, test


def create_model(width, height):
    input_shape = (width, height, 1)
    mn = 2 # max norm
    model = Sequential()
    model.add(Conv2D(
            128, kernel_size=(3, 3), kernel_constraint=max_norm(mn),
            activation='relu', kernel_initializer='he_uniform', input_shape=input_shape
        ))
    model.add(Conv2D(
        32, kernel_size=(3, 3), kernel_constraint=max_norm(mn), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2DTranspose(
        32, kernel_size=(3,3), kernel_constraint=max_norm(mn), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2DTranspose(
        128, kernel_size=(3,3), kernel_constraint=max_norm(mn), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(
        1, kernel_size=(3, 3), kernel_constraint=max_norm(mn), activation='sigmoid', padding='same'))
    # model.summary()
    return model


def train_model(pure_signal, noise_factor=0.1, width=8, height=5, epoch_cnt=10, batches=64, valid_split=0.7):
    noisy = add_noise(pure_signal, noise_factor=noise_factor)
    pure_r = reshape_signal(pure_signal, width, height)
    noisy_r = reshape_signal(noisy, width, height)
    
    pure_train, pure_test = data_split(pure_r, 1)
    noisy_train, noisy_test = data_split(noisy_r, 1)
    model = create_model(width,height)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(noisy_r, pure_r,
                    epochs=epoch_cnt,
                    batch_size=batches,
                    validation_split=valid_split)
    model.save("ae_denoise_model")
    return model


def denoise_signal(noisy_signals, model_fn="ae_denoise_model"):
    signal = reshape_signal(noisy_signals)
    model = tensorflow.keras.models.load_model(model_fn)
    cleared = model.predict(signal)
    
    cleared = np.array(cleared).reshape((len(noisy_signals), 8 * 5,))
    return cleared
    