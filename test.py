import csv
import pandas as pd
from glob import glob
import cv2

from os import walk

path = 'deploy/trainval/*/*_image.jpg'
      
files = glob(path)

for snapshot in self.files:
    bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    label = bbox[-2]

    