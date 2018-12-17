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

def predict_challenge(data_loader, model):
    """
    Runs the model inference on the test set and outputs the predictions
    """
    model_pred = np.array([])
    for i, X in enumerate(data_loader):
        if torch.cuda.is_available():
            X = X.cuda()
        output = model(X)
        predicted = predictions(output.data).cpu()
        predicted = predicted.numpy()
        model_pred = np.concatenate([model_pred, predicted])

        del X
    return model_pred

def main():
    # data loaders
    _, _, te_loader, get_semantic_label = get_train_val_test_loaders(num_classes=config('vgg.num_classes'))

    print('Load pretrained ...')

    # Load the pretrained model from pytorch
    model = models.inception_v3(pretrained = True)

    # Freeze training for all layers
    # for param in model.parameters():
    #     param.require_grad = False

    # num_features = model.classifier[6].in_features
    # features = list(model.classifier.children())[:-1] # Remove last layer
    # features.extend([nn.Linear(num_features, 23)]) # Add our layer with 23
    # model.classifier = nn.Sequential(*features) # Replace the model classifier

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    # Attempts to restore the latest checkpoint if exists
    model, _, _ = restore_checkpoint(model, config('vgg.checkpoint'))

    if torch.cuda.is_available():
        model = model.cuda()

    # Evaluate model
    files = glob('deploy/test/*/*_image.jpg')
    model_pred = predict_challenge(te_loader, model)

    print('saving challenge predictions...\n')

    links = [file.replace('deploy/test/', '') for file in files]
    links = [link.replace('_image.jpg', '') for link in links]

    # model_pred = [get_semantic_label(p) for p in model_pred]

    data = [[link, model_pred[idx]] for idx, link in enumerate(links)]

    pd_writer = pd.DataFrame(data, columns=['guid/image', 'label'])
    pd_writer.to_csv('result.csv', index=False, header=True)

if __name__ == '__main__':
    main()