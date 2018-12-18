from glob import glob
import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize
from torch.utils.data import Dataset, DataLoader
from utils import config
from PIL import Image
import csv
from torchvision import datasets, models, transforms
from imgaug import augmenters as iaa
import imgaug as ia

def get_train_val_test_loaders(num_classes):
    tr, va, te = get_train_val_dataset(num_classes=num_classes)
    
    batch_size = config('inception_v3.batch_size')
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader, te_loader, tr.get_semantic_label

def get_train_val_dataset(num_classes):
    tr = CarDataset('train', num_classes)
    va = CarDataset('val', num_classes)
    te = CarDataset('test', num_classes)
    return tr, va, te
        
class CarDataset(Dataset):

    def __init__(self, partition, num_classes):

        super().__init__()
        
        if partition not in ['train', 'val', 'test']:
            raise ValueError('Partition {} does not exist'.format(partition))
        
        self.partition = partition
        self.num_classes = num_classes
        
        if self.partition == 'test':
            path = 'deploy/test/*/*_image.jpg'
        else:
            path = 'deploy/trainval/*/*_image.jpg'

        self.files = glob(path)
        ind = int(len(self.files) * .8)
        if self.partition == 'train':
            self.files = self.files[:ind]
        elif self.partition == 'val':
            self.files = self.files[ind:]

        self.labels = []

        dic = {}

        if self.partition != 'test':
            with open('deploy/trainval/labels.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)

                for row in csv_reader:
                    dic[row['guid/image']] = row['label']

            for snapshot in self.files:
                file = snapshot.replace('deploy/trainval/', '').replace('_image.jpg', '')
                self.labels.append(int(dic[file]))

        self.labels = np.array(self.labels)
        self.labels[np.where(self.labels == 1)] = 0
        self.labels[np.where(self.labels == 2)] = 1

        # Apply image transformation for inception 
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print('Loading data...')
        # self.X = [tf(plt.imread(file)) for file in self.files]
        self.X = self.files
        self.y = self.labels

        print('Done loading data.')
        classes = [
            'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
            'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
            'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
            'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
            'Military', 'Commercial', 'Trains'
        ]

        labels = []

        with open('classes.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                labels.append(row['label'])

        numeric_label = np.arange(len(classes))

        self.semantic_labels = dict(zip(
            numeric_label,
            labels
        ))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.partition == 'test':
            return self.tf(plt.imread(self.X[idx]))
        return self.tf(plt.imread(self.X[idx])), torch.tensor(int(self.y[idx])).long()

    def get_semantic_label(self, numeric_label):
        """
        Returns the string representation of the numeric 2class label (e.g.,
        the numberic label 1 maps to the semantic label 'miniature_poodle').
        """
        return self.semantic_labels[numeric_label]

if __name__ == '__main__':
    ## Future note: check scipy imread and imresize
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

