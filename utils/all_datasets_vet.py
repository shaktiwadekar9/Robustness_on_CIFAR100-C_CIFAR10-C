import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

import glob
import os
from shutil import move
from os import rmdir


def load_cifar10(cfg):
    
    transform_train = transforms.Compose(
                        [
                        transforms.ToTensor(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.Normalize((0.49139968,  0.48215841,  0.44653091),
                                            (0.24703223,  0.24348513,  0.26158784))]
                        )
    

    transform_test = transforms.Compose(
                        [
                        transforms.ToTensor(),
                        transforms.Normalize((0.49139968,  0.48215841,  0.44653091),
                                            (0.24703223,  0.24348513,  0.26158784))]
                        )
    # Load training dataset
    dataset_train = torchvision.datasets.CIFAR10(root='./dataset',
                                            train=True,
                                            download=True,
                                            transform=transform_train)

    trainloader = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=cfg['trBatch'],
                                            shuffle=True,
                                            num_workers=cfg['nworkers']
                                            )


    # Load and create testloader using torchvision
    testset = torchvision.datasets.CIFAR10(root='./dataset',
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=cfg['trBatch'],
                                            shuffle=False,
                                            num_workers=cfg['nworkers'])

    valloader = torch.utils.data.DataLoader(testset,
                                            batch_size=cfg['trBatch'],
                                            shuffle=False,
                                            num_workers=cfg['nworkers'])
    
    return trainloader, valloader, testloader



def load_cifar100(cfg):
    
    # [0.50707516  0.48654887  0.44091784]
    # [0.26733429  0.25643846  0.27615047]
    transform_train = transforms.Compose(
                        [
                        transforms.ToTensor(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.Normalize((0.50707516,  0.48654887,  0.44091784),
                                            (0.26733429,  0.25643846,  0.27615047))]
                        )

    transform_test = transforms.Compose(
                        [
                        transforms.ToTensor(),
                        transforms.Normalize((0.50707516,  0.48654887,  0.44091784),
                                            (0.26733429,  0.25643846,  0.27615047))]
                        )
    
    # Load training dataset
    dataset_train = torchvision.datasets.CIFAR100(root='./dataset',
                                            train=True,
                                            download=True,
                                            transform=transform_train)

    trainloader = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=cfg['trBatch'],
                                            shuffle=True,
                                            num_workers=cfg['nworkers']
                                            )


    # Load and create testloader using torchvision
    testset = torchvision.datasets.CIFAR100(root='./dataset',
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=cfg['trBatch'],
                                            shuffle=False,
                                            num_workers=cfg['nworkers'])

    valloader = torch.utils.data.DataLoader(testset,
                                            batch_size=cfg['trBatch'],
                                            shuffle=False,
                                            num_workers=cfg['nworkers'])
    
    return trainloader, valloader, testloader

