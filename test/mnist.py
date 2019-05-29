# -*- coding: utf-8 -*-
"""
Test running autoencoders on mnist dataset
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt

from src import autoencoder as ae
from src import IO
#%% Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = True
N_TEST_IMG = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("default device:", device.type)



#%% LOAD DATA
data=IO.loaddata.loadmnist(DOWNLOAD_MNIST)

#%% Visualize data 

print(data.train_data.size())     # (60000, 28, 28)
print(data.train_labels.size())   # (60000)
plt.imshow(data.train_data[2].numpy(), cmap='gray')
plt.title('%i' % data.train_labels[2])
plt.show()

#%% Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

#%% Select autoencoder
autoencoder = ae.autoencoder_simple.AutoEncoder_v1().to(device)

#%% Load it into optimizer 
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)

#%% Define a loss function
loss_func = nn.MSELoss().to(device)

#%% Run autoencoder

#initialize figures
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot
loss_lowest=9e9
#plot original data in first row for comparison
view_data = data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.cpu().data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

#Start the training    
for epoch in range(EPOCH):
    for i, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, 28*28).to(device)   # batch x, shape (batch, 28*28)
        b_y = x.view(-1, 28*28).to(device)   # batch y, shape (batch, 28*28)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        t=loss.data
        if t< loss_lowest:
            loss_lowest-t
            encoder_best=encoded
            decoder_best=decoded

        if i % 100 == 0: #Every 100 iterations we compute the loss and plot a graphic comparison of our encoder
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy())
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.cpu().data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.show(); plt.pause(0.05)
plt.ioff()
plt.show()
