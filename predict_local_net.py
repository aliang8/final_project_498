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
    train_data = pd.read_csv('data/xyz_train.txt', header=None)
    valid_data = pd.read_csv('data/xyz_valid.txt', header=None)
    all_features = pd.concat((train_data.iloc[:, 1:-3], valid_data.iloc[:, 1:-3]))

    fmean=np.array(all_features.mean())
    fstd=np.array(all_features.std())


    # prep features and normalize
    test_data = pd.read_csv('output_t2.txt',header=0)
    features=[]
    for i in range(test_data.shape[0]):
        guid=test_data[:].values[i][0]
        cls=float(test_data[:].values[i][2])
        xmin=float(test_data[:].values[i][3])
        ymin=float(test_data[:].values[i][4])
        xmax=float(test_data[:].values[i][5])
        ymax=float(test_data[:].values[i][6])
        area=((xmax-xmin)*(ymax-ymin))
        proj = np.fromfile('deploy/test/'+str(guid)+'_proj.bin', dtype=np.float32)
        proj.resize([3, 4])
        p=proj.flatten()
        features.append([xmin,ymin,xmax,ymax,area,p[0],p[1],p[2],p[5],p[6]])

    features=np.array(features)
    features= (features - fmean) / fstd
    test_features = torch.Tensor(np.array(features))


    # Initialize network with trained weights
    model = LocalizationNet()
    model.load_state_dict(torch.load(config('local_net.checkpoint') + 'local_net.param'))
    model.eval()

    # Predict on test data
    prediction = model(test_features)

    # Write output to csv 
    with open('predictions.csv', 'w+') as pred:
        w = csv.writer(pred)

        # header
        w.writerow(['guid/image/axis','value'])

        for i in range(prediction.shape[0]):
            guid = test_data[:].values[i][0]
            if test_data[:].values[i][6]==-1:
                x=-1.851861
                y=-2.976852
                z=36.597186
            else:
                x=prediction.detach().numpy()[i,0]
                y=prediction.detach().numpy()[i,1]
                z=prediction.detach().numpy()[i,2]
            w.writerow([guid+'/x',x])
            w.writerow([guid+'/y',y])
            w.writerow([guid+'/z',z])

if __name__ == '__main__':
    main()