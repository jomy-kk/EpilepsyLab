# Chiang et al. (2019) Multiple noise types Autoencoders
# https://ieeexplore.ieee.org/document/8693790

import numpy as np
import pandas as pd
from scipy.signal import resample
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
#matplotlib.use('MACOSX')


def create_model(name):
    """
    Returns an autoenconder model as described by the authors.
    """

    model = tf.keras.Sequential(name=name)

    #Input
    model.add(tf.keras.layers.InputLayer(input_shape=(1024, 1), name='Input'))

    # Encoder

    model.add(tf.keras.layers.Conv1D(filters=40, kernel_size=(16, ), strides=2))
    model.add(tf.keras.layers.Activation(tf.keras.activations.elu))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(filters=20, kernel_size=(16, ), strides=2))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.elu))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(filters=20, kernel_size=(16, ), strides=2))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.elu))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(filters=20, kernel_size=(16, ), strides=2))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.elu))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(filters=40, kernel_size=(16, ), strides=2))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.elu))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=(16, ), strides=1))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.elu))
    model.add(tf.keras.layers.BatchNormalization())

    # Decoder

    model.add(tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=(16, ), strides=1))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.elu))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1DTranspose(filters=40, kernel_size=(16, ), strides=2))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.elu))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1DTranspose(filters=20, kernel_size=(16, ), strides=2))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.elu))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1DTranspose(filters=20, kernel_size=(16, ), strides=2))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.elu))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1DTranspose(filters=20, kernel_size=(16, ), strides=2))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.elu))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1DTranspose(filters=40, kernel_size=(16, ), strides=2))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.elu))
    model.add(tf.keras.layers.BatchNormalization())

    # Output
    model.add(tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=(15, ), strides=1, name='Output'))

    return model


def evaluate_model(model, weights, test_inputs, test_targets):
    """
    Evaluates a given model with the given weights. Returns the test accuracy and loss.
    Provide the following:
    :param model: A valid already trained classification model, such as a CNN.
    :param weights: Weight vectors of the model after training. Give the best.
    :param test_inputs: Input vectors reserved for testing.
    :param test_targets: Target vectors reserved for testing -- growth truth.
    """
    model.load_weights(weights)
    loss, acc = model.evaluate(test_inputs, test_targets)
    return loss, acc


def fit_model(model, train_inputs, train_targets,
                           loss_function='mse', optimizer='adam',
                           metrics=('mse', ), epochs=40, batch_size=32, patience=5, validation_split=.2,
                           verbose=True):

    model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(lr=3e-4), metrics=metrics)
    if(verbose):
        model.summary()

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_'+loss_function, patience=patience, verbose=verbose)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best.h5', monitor='val_'+loss_function, verbose=verbose, save_best_only=True)
    model_train = model.fit(train_inputs, train_targets, validation_split=validation_split,
                            callbacks=[earlystop, checkpoint], epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model_train


def fit_and_evaluate_model(model, train_inputs, train_targets, test_inputs, test_targets,
                           loss_function='mse', optimizer='adam',
                           metrics=('mse', ), epochs=40, batch_size=32, patience=5, validation_split=.2):
    """
    Trains a given model with the given optimization and loss functions for the given number of epochs.
    Stops based on the MSE metric. An early stop is defined with a convergence patience set by patience.
    A checkpoint is defined to save only the weights from the best model, based on the validation accuracy.
    A percentage of the training examples given by validation_split is used only for validation.
    It plots the training accuracy and loss and the validation accuracy and loss against the number of epochs,
    and evaluates the best model for the given dataset.
    """

    model_train = fit_model(model, train_inputs, train_targets,
                               loss_function=loss_function, optimizer=optimizer,
                               metrics=metrics, epochs=epochs, batch_size=batch_size,
                               patience=patience, validation_split=validation_split)

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(20, 7))

    loss_ax.set_title('Loss')
    loss_ax.plot(model_train.history['loss'], '-r', label='Train')
    loss_ax.plot(model_train.history['val_loss'], '-g', label='Validation')

    acc_ax.set_title(loss_function)
    acc_ax.plot(model_train.history[loss_function], '-r', label='Train')
    acc_ax.plot(model_train.history['val_'+loss_function], '-g', label='Validation')

    plt.legend(loc=4)
    plt.show()

    loss, acc = evaluate_model(model, 'best.h5', test_inputs, test_targets)

    print('\nAccuracy: {}'.format(acc))
    print('Loss: {}'.format(loss))


def min_max_norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))


def get_signal(filename):
    signal = ((pd.read_hdf(filename))['ECG']).to_numpy()
    sf = 1000  # in Hz
    n = len(signal)
    t = n / sf
    signal = resample(signal, int(360 * t))
    return signal



#################################################

signal = np.array([])
for filename in (
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-29 18-22-31__2021-03-29 19-23-20.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-29 19-23-20__2021-03-29 20-23-46.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-29 20-23-46__2021-03-29 21-24-21.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-29 21-24-21__2021-03-29 22-25-03.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-29 22-25-03__2021-03-30 00-01-10.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-30 00-01-10__2021-03-30 01-01-30.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-30 01-01-30__2021-03-30 02-01-36.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-30 02-01-36__2021-03-30 03-02-11.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-30 03-02-11__2021-03-30 04-35-28.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-30 04-35-28__2021-03-30 05-35-58.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-30 05-35-58__2021-03-30 06-36-00.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-30 06-36-00__2021-03-30 07-36-20.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-30 07-36-20__2021-03-30 08-36-39.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-30 08-36-39__2021-03-30 16-35-01.h5",
                "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-30 16-35-01__2021-03-30 17-35-07.h5",
                 "/Users/jomy/Desktop/sinais HSM/Bitalinosignal_2021-03-30 17-35-07__2021-03-30 18-37-18.h5",
                 ):
    signal = np.append(signal, get_signal(filename))

n = len(signal)
signal = signal[:int(n/1024)*1024]
sf = 360  # in Hz
n = len(signal)
t = n / sf

samples = np.linspace(0, t, n)
# pli = 20 * np.sin(2 * np.pi * 50 * samples)
bw = 10 * np.sin(2 * np.pi * 50 * samples)
noisy_signal = signal + bw

signal = min_max_norm(signal)
noisy_signal = min_max_norm(noisy_signal)

N = int(n/1024)

signal_samples = np.split(signal, N)
noisy_signal_samples = np.split(noisy_signal, N)

train_inputs = np.array(noisy_signal_samples[:-960])
train_targets = np.array(signal_samples[:-960])
test_inputs = np.array(noisy_signal_samples[len(noisy_signal_samples)-960:])
test_targets = np.array(signal_samples[len(signal_samples)-960:])

"""
plt.plot(train_inputs[5])
plt.show()
plt.plot(train_targets[5])
plt.show()
"""

train_inputs = np.expand_dims(train_inputs, -1)
train_targets = np.expand_dims(train_targets, -1)
test_inputs = np.expand_dims(test_inputs, -1)
test_targets = np.expand_dims(test_targets, -1)

print('# train samples:', train_inputs.shape)
print('# test samples:', test_inputs.shape)

model = create_model('draft')
fit_and_evaluate_model(model, train_inputs, train_targets, test_inputs, test_targets, 'mse', 'adam')

model.load_weights('best.h5')
denoised = model(np.expand_dims(test_inputs[0], 0))

plt.plot(test_inputs[0], 'red')  # noisy segment
plt.plot(denoised[0], 'green')  # denoised segment
plt.plot(test_targets[0], 'black')  # original segment
plt.show()

print('noisy mse =', (np.square(test_inputs[0] - test_targets[0])).mean())
print('denoised mse =', (np.square(denoised[0] - test_targets[0])).mean())

