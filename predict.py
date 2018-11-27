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

def predict_challenge(data_loader, model):
    """
    Runs the model inference on the test set and outputs the predictions
    """
    model_pred = np.array([])
    for i, (X, y) in enumerate(data_loader):
        if torch.cuda.is_available():
            X = X.cuda()
        output = model(X)
        predicted = predictions(output.data)
        predicted = predicted.numpy()
        model_pred = np.concatenate([model_pred, predicted])

        del X, y
    return model_pred

def main():
    composed = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # data loaders
    _, _, te_loader, get_semantic_label = get_train_val_test_loaders(
        composed, num_classes=config('vgg.num_classes'))

    print('Load pretrained ...')
    # Load the pretrained model from pytorch
    vgg16 = models.vgg16_bn(pretrained=True)
    vgg16.load_state_dict(torch.load("./vgg16_bn-6c64b313.pth"))
    print(vgg16.classifier[6].out_features) # 1000 

    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False

    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, 23)]) # 23 classes
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

    # Attempts to restore the latest checkpoint if exists
    model, _, _ = restore_checkpoint(vgg16, config('vgg.checkpoint'))

    if torch.cuda.is_available():
        model = model.cuda()

    # Evaluate model
    model_pred = predict_challenge(te_loader, model)

    print('saving challenge predictions...\n')
    model_pred = [get_semantic_label(p) for p in model_pred]
    pd_writer = pd.DataFrame(model_pred, columns=['predictions'])
    pd_writer.to_csv('result.csv', index=False, header=False)

if __name__ == '__main__':
    main()