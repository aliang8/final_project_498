"""
EECS 498 - Self Driving Cars
University of Michigan
Inference for localization task
"""

import numpy as np
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils
from utils import config

from local_net import LocalizationNet

def main():
    # read data
    tr = pd.read_csv('data/train_feats.txt', header=None)
    val = pd.read_csv('data/val_feats.txt', header=None)
    all_features = pd.concat((tr.iloc[:, 1: -3], val.iloc[:, 1: -3]))

    # prep features and normalize
    test = pd.read_csv('data/ssd_features.txt', header=0)
    features = []
    for i in range(test.shape[0]):
        guid = test[:].values[i][0]
        cls = float(test[:].values[i][2])
        xmin = float(test[:].values[i][3])
        ymin = float(test[:].values[i][4])
        xmax = float(test[:].values[i][5])
        ymax = float(test[:].values[i][6])
        area = ((xmax-xmin)*(ymax-ymin))
        proj = np.fromfile('deploy/test/'+str(guid)+'_proj.bin', dtype=np.float32)
        proj.resize([3, 4])
        p = proj.flatten()
        features.append([xmin,ymin,xmax,ymax,area,p[0],p[1],p[2],p[5],p[6]])

    features = np.array(features)
    features = (features - np.array(all_features.mean())) / np.array(all_features.std())
    test_features = torch.Tensor(np.array(features))


    # Initialize network with trained weights
    model = LocalizationNet()
    model.load_state_dict(torch.load(config('local_net.checkpoint') + 'local_net.param'))
    model.eval()

    # Predict on test data
    prediction = model(test_features)

    data = []

    # Write output to csv 
    for i in range(prediction.shape[0]):
        guid = test[:].values[i][0]
        if test[:].values[i][6]==-1:
            x=y=0
            z=40
        else:
            x, y, z = prediction.detach().numpy()[i]
        data.append([guid+'/x',x])
        data.append([guid+'/y',y])
        data.append([guid+'/z',z])

    pd_writer = pd.DataFrame(data, columns=['guid/image/axis', 'value'])
    pd_writer.to_csv('preds/pred_localization.csv', index=False, header=True)

if __name__ == '__main__':
    main()