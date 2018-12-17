#! /usr/bin/python3
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from train_common import *
import matplotlib.pyplot as plt
import time
import os
import copy
from dataset import get_train_val_test_loaders

from utils import config
import utils

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    for i, (X, y) in enumerate(data_loader):
        if use_gpu:
            X = X.cuda()
            y = y.cuda()

        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        del X, y

def _evaluate_epoch(tr_loader, va_loader, model, criterion, epoch, stats):
    """
    Evaluates the `model` on the train and validation set.
    """
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0.0, 0.0
        running_loss = []
        for X, y in tr_loader:
            if use_gpu:
                X = X.cuda()
                y = y.cuda()

            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())

            del X, y

        train_loss = np.mean(running_loss)
        train_acc = correct / total

    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in va_loader:
            if use_gpu:
                X = X.cuda()
                y = y.cuda()

            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())

            del X, y

        val_loss = np.mean(running_loss)
        val_acc = correct / total

    stats.append([val_acc, val_loss, train_acc, train_loss])
    print('Train acc: {}'.format(train_acc))
    print('Train loss: {}'.format(train_loss))
    print('Val acc: {}'.format(val_acc))
    print('Val loss: {}'.format(val_loss))

def main():

    classes = (
        'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
        'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
        'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
        'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
        'Military', 'Commercial', 'Trains'
    )

    # VGG-16 Takes 224x224 images as input, so we resize all of them
    composed = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    print('Data loaders')
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(composed, num_classes=config('vgg.num_classes'))

    print('Load pretrained ...')
    # Load the pretrained model from pytorch
    vgg16 = models.vgg16_bn(pretrained = True)
    vgg16.load_state_dict(torch.load("vgg16_bn-6c64b313.pth"))
    print(vgg16.classifier[6].out_features) # 1000 

    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False

    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, len(classes))]) # Add our layer with 4 outputs
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

    # Use GPU
    if use_gpu:
        vgg16.cuda() 
        
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print('Restore checkpoint...')
    model, start_epoch, stats = restore_checkpoint(vgg16, config('vgg.checkpoint'))

    # Evaluate the randomly initialized model
    _evaluate_epoch(tr_loader, va_loader, model, criterion, start_epoch,
        stats)

    save_checkpoint(model, start_epoch, config('vgg.checkpoint'), stats)
    
    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('vgg.num_epochs')):
        # Train model
        _train_epoch(tr_loader, model, criterion, optimizer)
        
        # Evaluate model
        _evaluate_epoch(tr_loader, va_loader, model, criterion, epoch+1, stats)

        # Save model parameters
        save_checkpoint(model, epoch+1, config('vgg.checkpoint'), stats)

if __name__ == '__main__':
    main()