# 3. Train the model

import torch
from torch.autograd import Variable
import torch.utils.data as loader
import numpy as np
from os import path, makedirs, listdir, getcwd

# Hyper Parameters
EPOCH = 100
LR = 0.0003
BATCH_SIZE = 128


def train(model, train_set, val_set, data):

    # Auxiliary function to save the model
    def save_model(save_name, optim, loss_f, lr, epoch=EPOCH):
        dir = './denoisingExperiments/ecg/woochan2018/Trained_Params/{}/{}_{}'.format(data.model, save_name, epoch)
        if not path.exists(dir):
            makedirs(dir)
        model.cpu()
        torch.save({'data_setting': data,
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer': optim,
                    'loss_function': loss_f,
                    'learning_rate': lr
                    },
                   dir + '/model.pth')
        np.save(dir + '/trainloss.npy', train_loss)
        np.save(dir + '/valloss.npy', val_loss)
        print("Step 3: Model Saved")


    cuda = True if torch.cuda.is_available() else False
    print(cuda)

    # Create tensors for training / validation
    train_set = torch.from_numpy(train_set).float()
    val_set = torch.from_numpy(val_set).float()

    loss_func = torch.nn.L1Loss()
    train_loss, val_loss = [], []
    if cuda:
        model.cuda()
        loss_func.cuda()

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)



    # Train the model
    try:
        # Generates mini_batchs for training. Loads data for validation.
        train_loader = loader.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
        v_x, v_y = Variable(val_set[:, 0:1, :, :]), Variable(val_set[:, 1:2, :, :])

        # Moves data and model to gpu if available
        if cuda:
            v_x, v_y = v_x.cuda(), v_y.cuda()

        print("Step 2: Model Training Start")

        for epoch in range(EPOCH):
            for step, train_data in enumerate(train_loader):
                b_x = Variable(train_data[:, 0:1, :, :]).cuda() if cuda else Variable(train_data[:, 0:1, :, :])
                b_y = Variable(train_data[:, 1:2, :, :]).cuda() if cuda else Variable(train_data[:, 1:2, :, :])

                de = model(b_x)
                loss = loss_func(de, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluates current model state on val data set
            pred = model(v_x)
            loss_val_set = loss_func(pred, v_y)
            print('Epoch: {} | train loss: {:.4f} | val loss: {:.4f}'.format(epoch + 1, loss.data.item(),
                                                                             loss_val_set.data.item()))
            train_loss.append(loss.data.item())
            val_loss.append(loss_val_set.data.item())

        print("Step 2: Model Training Finished")

        # Save trained Parameters
    except KeyboardInterrupt:

        if str(input("Save Parameters?(y/n): ")) == 'y':
            save_name = str(input("Save parameters as?: ")) + '_Interrupted'
            save_model(save_name, 'Adam', 'L1Loss', LR)
        else:
            print("Session Terminated. Parameters not saved")

    else:
        print("entering else statement")
        save_model('v4newdata1', 'Adam', 'L1Loss', LR)



