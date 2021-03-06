from denoising_experiments.ecg.woochan2018.read import *
from denoising_experiments.ecg.woochan2018.model import *
from denoising_experiments.ecg.woochan2018.train import *

if __name__ == '__main__':
    data, train_set, val_set = read_hsm()
    model = setup_model()
    train(model, train_set, val_set, data)
