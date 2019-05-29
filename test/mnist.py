# -*- coding: utf-8 -*-
"""
Test running autoencoders on mnist dataset
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

from src import autoencoder as ae
#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("default device:", device.type)

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5


# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,                        # download it if you don't have it
)

#%%

# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[2])
plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)



autoencoder = ae.autoencoder_simple.AutoEncoder_v1().to(device)
#autoencoder=autoencoder.cuda(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss().to(device)
#loss_func=loss_func.cuda(device)
#%%
# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
#view_data=view_data.cuda(device)
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.cpu().data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())
#%%
    
for epoch in range(EPOCH):
    for i, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, 28*28).to(device)   # batch x, shape (batch, 28*28)
        b_y = x.view(-1, 28*28).to(device)   # batch y, shape (batch, 28*28)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if i % 100 == 0:
#            Tensor.cpu(loss.data)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy())

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.cpu().data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.show(); plt.pause(0.05)

plt.ioff()
plt.show()

## visualize in 3D plot
#view_data = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
#encoded_data, _ = autoencoder(view_data)
#fig = plt.figure(2); ax = Axes3D(fig)
#X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
#values = train_data.train_labels[:200].numpy()
#for x, y, z, s in zip(X, Y, Z, values):
#    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
#ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
#plt.show()