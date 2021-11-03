# 1. Read and prepare data

import numpy as np
import matplotlib.pyplot as plt

from ioTools.user import *
from ioTools.ecg import ECGData, MITBIHData
from ioTools.emg import EMGData
from ioTools.acc import ACCData
from models.nn import NNDataPrep

# Used to create Input/Label data for NN models
class Data(MITBIHData, EMGData, ACCData, NNDataPrep):

    def __init__(self, model, motion, noiselevel):
        self.model = model
        self.motion = motion
        self.noiselevel = noiselevel
        MITBIHData.__init__(self, '/Users/jomy/OneDrive - Universidade de Lisboa/11º e 12º Semestres/EpilepsyLab/denoisingExperiments/ecg/woochan2018/mitdb')
        EMGData.__init__(self, '/Users/jomy/OneDrive - Universidade de Lisboa/11º e 12º Semestres/EpilepsyLab/denoisingExperiments/ecg/woochan2018/emgdata')
        ACCData.__init__(self, '/Users/jomy/OneDrive - Universidade de Lisboa/11º e 12º Semestres/EpilepsyLab/denoisingExperiments/ecg/woochan2018/accdata')
        NNDataPrep.__init__(self)

    def data_splitter(self, input_data, label_data, shuffle = True, ratio = 4):
        if np.shape(input_data) != np.shape(label_data):
            log("Input data and label data dimensions do not match.", 2)
        elif np.shape(input_data)[0] % (ratio + 1) != 0:
            log("Ratio({}:1) does not fit with input data dimension({})".format(ratio, np.shape(input_data)), 2)
        else:
            num_samples = np.shape(input_data)[0]
            tuple, train_set, test_set = [], [], []

            for sample in range(num_samples):
                tuple.append((input_data[sample], label_data[sample]))

            if shuffle == True:
                np.random.seed(0)
                rdn_idx = np.random.choice([1,0], size = (num_samples, ), p = [1-1./(ratio+1), 1./(ratio+1)])
                for i in range(num_samples):
                    if rdn_idx[i] == 1:
                        train_set.append(tuple[i])
                    else:
                        test_set.append(tuple[i])
            else:
                [train_set, test_set] = np.vsplit(tuple, [num_samples*(1-1./(ratio+1))])

            return np.array(train_set), np.array(test_set)


def read():
    noiselevel = 4
    data = Data('Convolutional Autoencoder', 'mixed', noiselevel = noiselevel)

    # Call data into numpy array format. Check source code for additional input specifications
    clean_ecg = data.get_all_ecg(tf=240000)  # Total of 14 recordings
    emg_noise = data.get_all_emg('woochan', tf=10000)  # 10,000 data points * 3 motions * 2 trials * 4 subjects
    acc_dat = data.get_all_acc('woochan', tf=10000)  # equiv to emg

    # Remove mean, normalize to range (-1,1), adjust for noiselevel setting.
    clean_ecg[0, :] -= np.mean(clean_ecg[0, :])
    clean_ecg[0, :] = clean_ecg[0, :] / max(abs(clean_ecg[0, :]))
    emg_noise[0, :] -= np.mean(emg_noise[0, :])
    emg_noise[0, :] = (emg_noise[0, :] / max(abs(emg_noise[0, :]))) * data.noiselevel
    for i in range(0, 3):
        acc_dat[i, :] -= np.mean(acc_dat[i, :])
        acc_dat[i, :] = (acc_dat[i, :] / max(abs(acc_dat[i, :]))) * float(data.noiselevel ** (0.5))

    # Repeat the emg noise segment to each ecg recording
    n_repeats = np.shape(clean_ecg)[1] / np.shape(emg_noise)[1]
    emg_noise = np.array(list(emg_noise.transpose()) * int(n_repeats)).transpose()
    acc_dat = np.array(list(acc_dat.transpose()) * int(n_repeats)).transpose()
    clean_acc = np.random.randn(np.shape(acc_dat)[0], np.shape(acc_dat)[1]) * 0.05  # N(0,0.05)

    # Generate noisy ECG by adding EMG noise
    noisy_ecg = clean_ecg + emg_noise

    # Plot example
    print("start plot")
    fig, (ax1, ax2) = plt.subplots(2, sharey=True)
    ax1.plot(noisy_ecg[0, 100:1000], color='k', linewidth=0.4, linestyle='-', label='train_set loss')
    ax1.legend(loc=2)
    ax1.set(title="Plot", ylabel='Train Loss')
    ax2.plot(clean_ecg[0, 100:1000], color='b', linewidth=0.4, linestyle='-', label='val_set loss')
    ax2.legend(loc=2)
    ax2.set(xlabel="Time", ylabel="Val Loss")
    plt.show()

    # Add ACC data onto clean/noisy ecg data
    input_dat = np.vstack((noisy_ecg, acc_dat))
    label_dat = np.vstack((clean_ecg, clean_acc))

    # Note Use of data_form = 2, which gives a 2D output for each training sample
    input_dat = data.reformat(input_dat, feature_len=300, data_form=2)
    label_dat = data.reformat(label_dat, feature_len=300, data_form=2)
    print("Input Data shape: {}".format(np.shape(input_dat)))
    print("Label Data shape: {}".format(np.shape(label_dat)))

    train_set, val_set = data.data_splitter(input_dat, label_dat, shuffle=True, ratio=4)

    print("Step 0: Data Import Done")

    return data, train_set, val_set




class HSMData(ECGData, EMGData, ACCData, NNDataPrep):

    def __init__(self, model, motion, noiselevel):
        self.model = model
        self.motion = motion
        self.noiselevel = noiselevel
        ECGData.__init__(self, '/Users/jomy/Desktop/sinais HSM')
        EMGData.__init__(self, '/Users/jomy/OneDrive - Universidade de Lisboa/11º e 12º Semestres/EpilepsyLab/denoisingExperiments/ecg/woochan2018/emgdata')
        ACCData.__init__(self, '/Users/jomy/OneDrive - Universidade de Lisboa/11º e 12º Semestres/EpilepsyLab/denoisingExperiments/ecg/woochan2018/accdata')
        NNDataPrep.__init__(self)

    def data_splitter(self, input_data, label_data, shuffle = True, ratio = 4):
        if np.shape(input_data) != np.shape(label_data):
            log("Input data and label data dimensions do not match.", 2)
        elif np.shape(input_data)[0] % (ratio + 1) != 0:
            log("Ratio({}:1) does not fit with input data dimension({})".format(ratio, np.shape(input_data)), 2)
        else:
            num_samples = np.shape(input_data)[0]
            tuple, train_set, test_set = [], [], []

            for sample in range(num_samples):
                tuple.append((input_data[sample], label_data[sample]))

            if shuffle == True:
                np.random.seed(0)
                rdn_idx = np.random.choice([1,0], size = (num_samples, ), p = [1-1./(ratio+1), 1./(ratio+1)])
                for i in range(num_samples):
                    if rdn_idx[i] == 1:
                        train_set.append(tuple[i])
                    else:
                        test_set.append(tuple[i])
            else:
                [train_set, test_set] = np.vsplit(tuple, [num_samples*(1-1./(ratio+1))])

            return np.array(train_set), np.array(test_set)


def read_hsm():
    noiselevel = 4
    data = HSMData('Convolutional Autoencoder', 'mixed', noiselevel = noiselevel)

    # Call data into numpy array format. Check source code for additional input specifications
    clean_ecg = data.get_all_ecg('hsm', tf = 960000)
    emg_noise = data.get_all_emg('woochan', tf=10000)  # 10,000 data points * 3 motions * 2 trials * 4 subjects
    acc_dat = data.get_all_acc('woochan', tf=10000)  # equiv to emg

    print(np.shape(clean_ecg))
    print(np.shape(emg_noise))
    print(np.shape(acc_dat))

    # Remove mean, normalize to range (-1,1), adjust for noiselevel setting.
    clean_ecg[0, :] -= np.mean(clean_ecg[0, :])
    clean_ecg[0, :] = clean_ecg[0, :] / max(abs(clean_ecg[0, :]))
    emg_noise[0, :] -= np.mean(emg_noise[0, :])
    emg_noise[0, :] = (emg_noise[0, :] / max(abs(emg_noise[0, :]))) * data.noiselevel
    for i in range(0, 3):
        acc_dat[i, :] -= np.mean(acc_dat[i, :])
        acc_dat[i, :] = (acc_dat[i, :] / max(abs(acc_dat[i, :]))) * float(data.noiselevel ** (0.5))

    # Repeat the emg noise segment to each ecg recording
    n_repeats = np.shape(clean_ecg)[1] / np.shape(emg_noise)[1]
    emg_noise = np.array(list(emg_noise.transpose()) * int(n_repeats)).transpose()
    acc_dat = np.array(list(acc_dat.transpose()) * int(n_repeats)).transpose()
    clean_acc = np.random.randn(np.shape(acc_dat)[0], np.shape(acc_dat)[1]) * 0.05  # N(0,0.05)

    # Generate noisy ECG by adding EMG noise
    noisy_ecg = clean_ecg + emg_noise

    # Plot example
    print("start plot")
    fig, (ax1, ax2) = plt.subplots(2, sharey=True)
    ax1.plot(noisy_ecg[0, 100:1000], color='k', linewidth=0.4, linestyle='-', label='train_set loss')
    ax1.legend(loc=2)
    ax1.set(title="Plot", ylabel='Train Loss')
    ax2.plot(clean_ecg[0, 100:1000], color='b', linewidth=0.4, linestyle='-', label='val_set loss')
    ax2.legend(loc=2)
    ax2.set(xlabel="Time", ylabel="Val Loss")
    plt.show()

    # Add ACC data onto clean/noisy ecg data
    input_dat = np.vstack((noisy_ecg, acc_dat))
    label_dat = np.vstack((clean_ecg, clean_acc))

    # Note Use of data_form = 2, which gives a 2D output for each training sample
    input_dat = data.reformat(input_dat, feature_len=300, data_form=2)
    label_dat = data.reformat(label_dat, feature_len=300, data_form=2)
    print("Input Data shape: {}".format(np.shape(input_dat)))
    print("Label Data shape: {}".format(np.shape(label_dat)))

    train_set, val_set = data.data_splitter(input_dat, label_dat, shuffle=True, ratio=4)

    print("Step 0: Data Import Done")

    return data, train_set, val_set











