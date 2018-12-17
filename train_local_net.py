import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils
from utils import config

from local_net import LocalizationNet

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

def rmse(model, X, y):
	prediction = model(X)
	loss = nn.MSELoss()
	rmse = np.sqrt(2 * loss(prediction, y).detach().mean())
	return rmse.item()

def train(model, tr_loader, train_X, train_y, val_X, val_y):
	# Loss function and optimizer
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=config('local_net.learning_rate'), weight_decay=config('local_net.weight_decay'))

	num_epochs = config('local_net.num_epochs')
	for epoch in range(num_epochs):
		for X, y in tr_loader:
			if use_gpu:
				X = X.cuda()
				y = y.cuda()

			# clear parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			output = model(X)
			loss = criterion(output, y)
			loss.backward()
			optimizer.step()

		train_loss = rmse(model, train_X, train_y)
		val_loss = rmse(model, val_X, val_y)
		print("Epoch {}, train_loss = {}, val_loss = {}".format(epoch, train_loss, val_loss))
	
def main():
	model = LocalizationNet()

	# Read data 
	train_data = pd.read_csv('data/xyz_train.txt', header=None)
	valid_data = pd.read_csv('data/xyz_valid.txt', header=None)

	all_features = pd.concat((train_data.iloc[:, 1: -3], valid_data.iloc[:, 1: -3]))
	all_labels = pd.concat((train_data.iloc[:, -3:], valid_data.iloc[:, -3:]))
	numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
	all_features[numeric_features] = all_features[numeric_features].apply(
		lambda x: (x - x.mean()) / (x.std()))

	n_train = train_data.shape[0]
	train_X = np.array(all_features[:n_train].values)
	val_X = np.array(all_features[n_train:].values)
	train_y = np.array(all_labels[:n_train].values)
	val_y = np.array(all_labels[n_train:].values)

	# Create dataset 
	train_X = torch.stack([torch.Tensor(i) for i in train_X]) 
	train_y = torch.stack([torch.Tensor(i) for i in train_y])
	
	ds = utils.TensorDataset(train_X,train_y) 
	tr_loader = utils.DataLoader(ds, config('local_net.batch_size'), shuffle=True) 

	val_X = torch.stack([torch.Tensor(i) for i in val_X]) 
	val_y = torch.stack([torch.Tensor(i) for i in val_y])

	# Train model
	train(model, tr_loader, train_X, train_y, val_X, val_y)

	# Save model weights
	torch.save(model.state_dict(), config('local_net.checkpoint') + 'local_net.param')

if __name__ == '__main__':
	main()
