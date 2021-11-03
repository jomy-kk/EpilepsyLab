###################################

# EpilepsyLAB

# Package: io
# File: ecg
# Description: Procedures to read and write ECG datafiles.

# Contributors: João Saraiva, Mariana Abreu
# Last update: 01/11/2021

###################################

from os import getcwd, listdir, path
import wfdb
import pandas as pd
import numpy as np
from scipy.signal import resample

from io_tools.user import *



class Noise:
    """
    Object to characterize noises of an arbitrary ECG signal.
    """

    def __init__(self):
        pass

    # Set parameters for guassian noise
    def set_gaussian_noise(self, signal, amp=1, u=0, std=0.01):
        self.gaussian = np.multiply(amp, np.random.normal(u, std, len(signal)))

    # Set parameters for baseline wander
    def set_baseline_wander(self, signal, respf=12, amp=0.05):
        N = float(len(signal))
        resppm = (float(respf) / 60)
        ix = np.arange(N)
        self.baseline_wander = amp * np.sin(2 * np.pi / (resppm * (N / 2)) * ix)


class ECGData:
    """
    Object to open and keep ECG data from multiple sources.
    """

    def __init__(self, directory:str=None):
        if directory is None:
            self.ecg_path = getcwd()
        else:
            self.set_ecg_path(directory)

        log("ECGs will be imported from {}".format(self.ecg_path))


    def set_ecg_path(self, directory:str):
        if path.isdir(directory):
            self.ecg_path = directory
        else:
            log("Directory {} does not exist.".format(directory), 2)


    def get_as_h5(self):
        def get_signal(filename):
            log("Collecting {}...".format(filename))
            signal = ((pd.read_hdf(path.join(self.ecg_path, filename)))['ECG']).to_numpy()
            sf = 1000  # in Hz
            n = len(signal)
            t = n / sf
            signal = resample(signal, int(360 * t))
            return signal

        filepaths = [file for file in sorted(listdir(self.ecg_path)) if file.endswith('.h5')]
        if filepaths == []:
            log("There are no H5 files in {}".format(self.ecg_path), 2)
        else:
            self.data = np.array([])
            for file in filepaths:
                self.data = np.append(self.data, get_signal(file))

            return self.data


    def get_all_ecg(self, source, tf):
        if source == 'hsm':
            self.get_as_h5()
            self.data = self.data[:tf]
            self.data = np.reshape(self.data, (1, np.shape(self.data)[0]))
            return self.data
        else:
            log("Specify a valid source. Current acceptable sources are 'hsm' (hospital de santa maria).", 2)


    def set_noise(self, noise:Noise):
        self.noise = noise

    def apply_noise(self):
        return self.data + self.noise.gaussian + self.noise.baseline_wander


class MITBIHData(ECGData):
    """
    Object to open and keep ECG data from the MIT-BIH arrhythmia database
    """

    def __init__(self, directory: str = None):
        download = False

        if directory is None:
            answer = query("No ecg_path specified. Download the MIT-BIH arrhythmia database to the current ecg_path?", yesorno=True)
            if answer:
                download = True
            else:
                log("Exiting.")
                exit(0)

        else:
            if path.isdir(directory):
                self.ecg_path = directory
            else:
                answer = query("Directory {} does not exist. Download the MIT-BIH arrhythmia database to this ecg_path, creating it?".format(directory), yesorno=True)
                if answer:
                    download = True
                else:
                    log("Exiting.")
                    exit(0)

        if download:
            pass
            # TODO: download and save

        super().__init__(directory)


    # Pull all data using wfdb format. Use pull_signal to get raw data.
    def get_wfdb(self, filepath, filename, t0 = 0, tf = int(30 * 360 * 60)):
        return wfdb.rdsamp("{}/{}".format(filepath, filename), sampfrom = t0, sampto = tf, channels = [0])

    # Search and extract all data in the file_path using wfdb format. Use pull_all_signal to extract signal.
    def get_all_wfdb(self, t0 = 0, tf = int(30 * 360 * 60)):
        items = listdir(self.ecg_path)
        items.sort()
        newlist = []
        namelist = []
        for name in items:
            if name.endswith(".dat"):
                namelist.append(name)
                name = name[:-4]
                dat = wfdb.rdsamp("{}/{}".format(self.ecg_path, name), sampfrom = t0, sampto = tf, channels = [0])
                newlist.append(dat)
        print('These are the files[ECG] opened from the dir: {}'.format(namelist))
        return newlist

    # Save raw ecg signal from the wfdb format. 30 min each. 360Hz. 11bit. 10mV.
    def get_ecg(self, filename, t0 = 0, tf = int(20 * 360 * 60)):
        temp_data = self.get_wfdb(filename, t0, tf)
        if type(temp_data) == tuple:
            signal = temp_data[0]
        else:
            signal = temp_data.p_signals
        out_data = np.reshape(signal,(1, tf-t0))
        return out_data

    # Output raw ecg signal from all .dat files in file_path.
    def get_all_ecg(self, t0 = 0, tf = int(30 * 360 * 60)):
        output = []
        for temp_data in self.get_all_wfdb(t0, tf):
            if type(temp_data) == tuple:
                signal = temp_data[0]
            else:
                signal = temp_data.p_signals
            output.append(signal)
        output = np.array(output)
        output = np.reshape(output, (1, np.shape(output)[0]*np.shape(output)[1]))
#        print('Created signal(numpy array) with shape: {}'.format(np.shape(output)))
        self.data = output
        return output




