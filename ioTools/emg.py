###################################

# EpilepsyLAB

# Package: io
# File: emg
# Description: Procedures to read and write EMG datafiles.

# Contributors: João Saraiva
# Last update: 01/11/2021

###################################

from os import getcwd, listdir, path
import numpy as np

from ioTools.user import *


class EMGData:
    """
    Object to open and keep EMG data from multiple sources.
    """

    def __init__(self, directory:str=None):
        if directory is None:
            self.emg_path = getcwd()
        else:
            self.set_emg_path(directory)

        log("EMGs will be imported from {}".format(self.emg_path))


    def set_emg_path(self, directory:str):
        if path.isdir(directory):
            self.emg_path = directory
        else:
            log("Directory {} does not exist.".format(directory), 2)


    def get_emg_as_csv(self, filename, t0, tf):
        return np.genfromtxt("{}/{}.csv".format(self.emg_path, filename), delimiter=',')[t0:tf]


    def get_all_emg(self, source, tf):
        if source == 'woochan':
            motionlist = ['motion1', 'motion3', 'motion4']
            newlist = []
            # Open each motion file and all data files within each. Concat all to newlist
            for motion in motionlist:
                motion_path = self.emg_path + '/' + motion
                items = listdir(motion_path)
                items.sort()
                for name in items:
                    if name.endswith(".csv"):
                        name = name[:-4]
                        data = self.get_emg_as_csv(filename=motion + '/' + name, t0=1, tf=tf + 1)[:, 1]
                        if len(data) != tf:
                            print("Not enough data: ", len(data))
                        newlist.append(data)
                print("EMG: Loaded {}".format(motion))
            signal = np.reshape(np.array(newlist), (1, -1))
            return signal

        else:
            log("Specify a valid EMG source. Current acceptable sources are 'woochan'.", 2)

