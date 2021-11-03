###################################

# EpilepsyLAB

# Package: io
# File: acc
# Description: Procedures to read and write accelerometer datafiles.

# Contributors: João Saraiva
# Last update: 01/11/2021

###################################

from os import getcwd, listdir, path
import numpy as np

from ioTools.user import *


class ACCData:
    """
    Object to open and keep accelerometer data from multiple sources.
    """

    def __init__(self, directory: str = None):
        if directory is None:
            self.acc_path = getcwd()
        else:
            self.set_acc_path(directory)

        log("ACCs will be imported from {}".format(self.acc_path))


    def set_acc_path(self, directory:str):
        if path.isdir(directory):
            self.acc_path = directory
        else:
            log("Directory {} does not exist.".format(directory), 2)


    def get_acc_as_csv(self, filename, t0, tf):
        return np.genfromtxt("{}/{}.csv".format(self.acc_path, filename), delimiter = ',')[:,t0:tf]

    def get_all_acc(self, source, tf):
        if source == 'woochan':
            motionlist = ['motion1', 'motion3', 'motion4']
            newlist = []
            c = 0
            # Open each motion file and all data files within each. Concat all to newlist
            for motion in motionlist:
                motion_path = self.acc_path + '/' + motion
                items = listdir(motion_path)
                items.sort()
                for name in items:
                    if name.endswith(".csv"):
                        name = name[:-4]
                        data = self.get_acc_as_csv(filename=motion + '/' + name, t0=0, tf=tf)
                        if len(data) != tf:
                            print("Not enough data: ", len(data))
                        newlist.append(data)
                        c += 1
                print("ACC: Loaded {}".format(motion))
            self.opened_acc = c
            signal = np.reshape(np.array(newlist), (-1, 3)).transpose()
            return signal

        else:
            log("Specify a valid ACC source. Current acceptable sources are 'woochan'.", 2)

