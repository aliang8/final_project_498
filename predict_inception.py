"""
EECS 498 - Self Driving Cars
University of Michigan
Inference for classification task
"""

import argparse
import torch
import numpy as np
import pandas as pd
import utils
from dataset import get_train_val_test_loaders
from train_common import *
from utils import config
from torchvision import datasets, models, transforms
import torch.nn as nn
from glob import glob

def predict(data_loader_1, data_loader_2, model_1, model_2):
    """
    Runs the model inference on the test set and outputs the predictions
    """
    model_pred = np.array([])
    for X_1, X_2 in zip(data_loader_1, data_loader_2):
        if torch.cuda.is_available():
            X_1 = X_1.cuda()
            X_2 = X_2.cuda()
        output_1 = model_1(X_1)
        output_2 = model_2(X_2)
        predicted_1 = predictions(output_1[0].data).cpu().numpy()
        predicted_2 = predictions(output_2.data).cpu().numpy()
        predicted = np.round(.7 * predicted_1 + .3 * predicted_2)
        model_pred = np.concatenate([model_pred, predicted])

        del X_1, X_2
    return model_pred

def main():
    # data loaders
    _, _, te_loader_inception, _ = get_train_val_test_loaders('inception_v3', num_classes=config('inception_v3.num_classes'))

    _, _, te_loader_resnet, _ = get_train_val_test_loaders('resnet18', num_classes=config('resnet18.num_classes'))

    # Ensemble model~

    # Load the pretrained model for inception
    model_inception = models.inception_v3(pretrained = True)
    num_ftrs = model_inception.fc.in_features
    model_inception.fc = torch.nn.Linear(num_ftrs, 2)

    model_resnet = models.resnet18(pretrained = True)
    num_ftrs = model_resnet.fc.in_features
    model_resnet.fc = torch.nn.Linear(num_ftrs, 2)


    # Attempts to restore the latest checkpoint if exists
    model_inception, _, _ = restore_checkpoint(model_inception, config('inception_v3.checkpoint'))
    model_resnet, _, _ = restore_checkpoint(model_resnet, config('resnet18.checkpoint'))

    if torch.cuda.is_available():
        model_inception = model_inception.cuda()
        model_resnet = model_resnet.cuda()

    # Evaluate model
    files = glob('deploy/test/*/*_image.jpg')

    model_pred = predict(te_loader_inception, te_loader_resnet, model_inception, model_resnet)
    model_pred[np.where(model_pred == 1)] = int(2)
    model_pred[np.where(model_pred == 0)] = int(1)

    print('saving challenge predictions...\n')

    # Write to prediction file
    links = [file.replace('deploy/test/', '') for file in files]
    links = [link.replace('_image.jpg', '') for link in links]
    data = [[link, model_pred[idx]] for idx, link in enumerate(links)]

    pd_writer = pd.DataFrame(data, columns=['guid/image', 'label'])
    pd_writer.to_csv('preds/pred_classification.csv', index=False, header=True)

if __name__ == '__main__':
    main()