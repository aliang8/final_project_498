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

def get_train_val_test_loaders(transforms, num_classes):
    tr, va, te = get_train_val_dataset(transforms, num_classes=num_classes)
    
    batch_size = config('vgg.batch_size')
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader, te_loader, tr.get_semantic_label

def get_train_val_dataset(transforms, num_classes):
    tr = CarDataset('train', transforms, num_classes)
    va = CarDataset('val', transforms, num_classes)
    te = CarDataset('test', transforms, num_classes)

    return tr, va, te
        
class CarDataset(Dataset):

    def __init__(self, partition, transforms, num_classes):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()
        
        if partition not in ['train', 'val', 'test']:
            raise ValueError('Partition {} does not exist'.format(partition))
        
        np.random.seed(0)
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
        for snapshot in self.files:
            try:
                bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
                label = bbox[-2]
                self.labels.append(label)
            except FileNotFoundError:
                bbox = np.array([], dtype=np.float32)
                self.labels.append('')

        classes = [
            'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
            'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
            'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
            'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
            'Military', 'Commercial', 'Trains'
        ]

        numeric_label = np.arange(len(classes))

        self.semantic_labels = dict(zip(
            numeric_label,
            classes
        ))

        self.transforms = transforms

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        if label != '':
            label = int(self.labels[idx])
        
        img_as_tensor = plt.imread(self.files[idx])

        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_tensor)

        # Return image and the label
        return (img_as_tensor, label)

    def get_semantic_label(self, numeric_label):
        """
        Returns the string representation of the numeric class label (e.g.,
        the numberic label 1 maps to the semantic label 'miniature_poodle').
        """
        return self.semantic_labels[numeric_label]

if __name__ == '__main__':
    ## Future note: check scipy imread and imresize
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    np.set_printoptions(precision=3)
    tr, va, te, standardizer = get_train_val_dataset()
    print("Train:\t", len(tr.X))
    print("Val:\t", len(va.X))
    print("Test:\t", len(te.X))
