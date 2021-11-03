###################################

# EpilepsyLAB

# Package: models
# File: nn
# Description: Ready-to-use skeletons of neural network models, and
#              procedures to manipulate data for them.

# Contributors: João Saraiva
# Last update: 01/11/2021

###################################

import numpy as np
import torch


class NNDataPrep:
    """
    Data arrangements for neural networks.
    """

    # Change data format to specified form / length
    # Format 1: <ECG..., X..., Y..., Z...>; each * feature_len
    # Total lenght being 4 * feature_len
    # Format 2: <Sample num, 4, feature_len>
    def reformat(self, data, data_form=0, feature_len=0):
        self.format = data_form
        self.feature_len = feature_len

        if data_form == 0 or feature_len == 0:
            print("Please specify format type and length")
        else:
            if data_form == 1:
                output = self.format1(data, feature_len)
            elif data_form == 2:
                output = self.format2(data, feature_len)
            else:
                print("Undefined format type")
        return output

    # Revert output to numpy of original format. Accepts numpy not tensor object
    # To be used to plot denoised results
    def undo_reformat(self, data):
        if self.format == 1:
            output = self.undo_format1(data, self.feature_len)
        elif self.format == 2:
            output = self.undo_format2(data, self.feature_len)
        else:
            print('Unalbe to undo format')
        return output

    # These functions should not be called directly
    def format1(self, input_arr, feature_len):
        l = int(np.shape(input_arr)[1])
        k = int(feature_len)
        sample_num = int(l / feature_len)
        output = np.zeros((sample_num, 4 * k))
        for i in range(0, sample_num):
            output[i, 0:k] = input_arr[0, k * i:k * (i + 1)]
            output[i, k:2 * k] = input_arr[1, k * i:k * (i + 1)]
            output[i, 2 * k:3 * k] = input_arr[2, k * i:k * (i + 1)]
            output[i, 3 * k:4 * k] = input_arr[3, k * i:k * (i + 1)]
        return np.array(output)

    def undo_format1(self, npver, feature_len):
        k = int(feature_len)
        sample_num = np.shape(npver)[0]
        sig_len = sample_num * k
        output = np.zeros((4, sig_len))
        for i in range(0, sample_num):
            output[0, k * i:k * (i + 1)] = npver[i, 0:k]
            output[1, k * i:k * (i + 1)] = npver[i, k:2 * k]
            output[2, k * i:k * (i + 1)] = npver[i, 2 * k:3 * k]
            output[3, k * i:k * (i + 1)] = npver[i, 3 * k:4 * k]
        return np.array(output)

    def format2(self, input_arr, feature_len):
        l = int(np.shape(input_arr)[1])
        k = int(feature_len)
        sample_num = int(l / feature_len)
        output = np.zeros((sample_num, 4, k))
        for i in range(0, int(sample_num)):
            output[i] = input_arr[:, k * (i):k * (i + 1)]
        return np.array(output)

    def undo_format2(self, npver, feature_len):
        k = int(feature_len)
        sample_num = np.shape(npver)[0]
        sig_len = sample_num * k
        if np.shape(npver)[2] == 4:
            output = np.zeros((4, sample_num * k))
            for i in range(1, sample_num):
                output[:, k * i:k * (i + 1)] = npver[i]
        elif np.shape(npver)[2] == 1:
            output = np.zeros((1, sample_num * k))
            for i in range(1, sample_num):
                output[:, k * i:k * (i + 1)] = npver[i]
        return np.array(output)

    # Formating into tensors usable in PyTorch. This is currently NOT wrapped in a Variable
    def to_tensor(self, data):
        return torch.from_numpy(data).float()

    def to_numpy(self, tensor):
        return tensor.data.numpy()
