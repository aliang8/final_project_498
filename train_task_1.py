"""
EECS 498 - Self Driving Cars
University of Michigan
Main file for training ensemble model
"""

from train_inception import train_inception
from train_resnet import train_resnet

if __name__ == '__main__':
	print('Training inception network for 10 epochs.........')
	train_inception()
	
	print('Training resnet network for 10 epochs.........')
	train_resnet()