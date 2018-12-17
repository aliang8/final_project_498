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
        loss = criterion(output[0], y)
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

            output = model(X)[0]
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
            # print(running_loss)
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

            output = model(X)[0]
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
    print('Data loaders')
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(num_classes=config('inception_v3.num_classes'))

    print('Load pretrained ...')

    # Load the pretrained model from pytorch
    model = torchvision.models.inception_v3(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
        
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config('inception_v3.learning_rate'), weight_decay=0.0005)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # # scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print('Restore checkpoint...')
    model, start_epoch, stats = restore_checkpoint(model, config('inception_v3.checkpoint'))

    # Use GPU
    if use_gpu:
        model.cuda() 

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('inception_v3.num_epochs')):

        # scheduler.step()

        print('Training Epoch: {}'.format(epoch))
        start = time.time()

        # Train model
        _train_epoch(tr_loader, model, criterion, optimizer)

        print('Time for training: {}'.format(time.time() - start))


        print('Evaluating Epoch: {}'.format(epoch))        
        start = time.time()

        # Evaluate model
        _evaluate_epoch(tr_loader, va_loader, model, criterion, epoch+1, stats)
        print('Time for evaluating: {}'.format(time.time() - start))

        # Save model parameters
        save_checkpoint(model, epoch+1, config('inception_v3.checkpoint'), stats)

if __name__ == '__main__':
    main()