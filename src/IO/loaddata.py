# -*- coding: utf-8 -*-
"""
Data loader 
Contains loadmnist
"""
import torchvision

def loadmnist(DOWNLOAD_MNIST):
    train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     
    transform=torchvision.transforms.ToTensor(),                                                
    download=DOWNLOAD_MNIST,                        )
    return train_data
