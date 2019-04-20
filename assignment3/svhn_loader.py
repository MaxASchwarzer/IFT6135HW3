# Dependencies
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import random

import os
import sys


# Define a function to sanitize the .mat files and re-save
def sanitize_mat(filename):

	"""
	inputs :

	filename :
		The path to the .mat file

	outputs :
	"""

	mat = sio.loadmat(filename)

	data = mat["X"]
	label = mat["y"].astype(np.int64).squeeze()

	# Replace class label 10 with class label 0 to comply with pytorch.
	# If we have C classes, the class label range from 0 to C-1 and not
	# 1 to C
	np.place(label, label == 10, 0)
	# PyTorch/Torch uses data of shape: Batch x Channel x Height x Width
	# SVHN data is in shape: Height x Width x Channel x Batch.
	data = np.transpose(data, (3, 2, 0, 1))

	sio.savemat(filename, {"X": data, "y": label})


# Define a function to create the train-valid splits from the train file
def create_train_val_split(split=0.20):
	"""
	Create a train and validation split and save into separate files

	Args:
		split (int): The percentage of data to be in the validation set
						from the test set
	"""
	train_mat = sio.loadmat('train_32x32.mat')

	data = train_mat["X"]
	label = train_mat["y"]
	# label = train_mat["y"].astype(np.int64).squeeze()
	# # Replace class label 10 with class label 0 to comply
	# np.place(label, svhn_label == 10, 0)
	# data = np.transpose(data, (3, 2, 0, 1))

	# Shuffle
	combined = list(zip(data, label))
	random.shuffle(combined)
	data[:], label[:] = zip(*combined)

	# Split
	split_idx = int(data.shape[0]*split)
	train_data = data[split_idx:]
	train_label = label[split_idx:]
	val_data = data[:split_idx]
	val_label = label[:split_idx]

	sio.savemat("train_split_32x32.mat", {"X": train_data, "y": train_label})
	sio.savemat("val_split_32x32.mat", {"X": val_data, "y": val_label})


class TestDataset(Dataset):
	"""SVHN Test dataset"""

	def __init__(self, mat_file_loc, transform=None):
		"""
		Args:
			mat_file_loc (string): Path to the mat file containing test data
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.test_mat = sio.loadmat(mat_file_loc)
		self.data = self.test_mat["X"]
		self.label = self.train_mat["y"]
		# self.data = np.transpose(self.test_mat["X"], (3, 2, 0, 1))
		self.transform = transform

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		return self.transform(self.data[idx]), self.label[idx]


class TrainDataset(Dataset):
	"""SVHN Train dataset"""

	def __init__(self, mat_file_loc, transform=None):
		"""
		Args:
			mat_file_loc (string): Path to the mat file containing train data
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.train_mat = sio.loadmat(mat_file_loc)
		self.data = self.train_mat["X"]
		self.label = self.train_mat["y"]
		# self.data = np.transpose(self.train_mat["X"], (3, 2, 0, 1))
		# self.label = self.train_mat["y"].astype(np.int64).squeeze()
		# self.label = np.place(self.label, self.label == 10, 0)
		self.transform = transform

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		# return {"data": self.transform(self.data[idx]), "label": self.label[idx]}
		return self.transform(self.data[idx]), self.label[idx]


class ValidDataset(Dataset):
	"""SVHN Validation dataset"""

	def __init__(self, mat_file_loc, transform=None):
		"""
		Args:
			mat_file_loc (string): Path to the mat file containing validation data
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.val_mat = sio.loadmat(mat_file_loc)
		self.data = self.val_mat["X"]
		self.label = self.val_mat["y"]
		# self.data = np.transpose(self.val_mat["X"], (3, 2, 0, 1))
		# self.label = self.val_mat["y"].astype(np.int64).squeeze()
		# self.label = np.place(self.label, self.label == 10, 0)
		self.transform = transform

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		# return {"data": self.transform(self.data[idx]), "label": self.label[idx]}
		return self.transform(self.data[idx]), self.label[idx]


# Define a method to load the standard SVHN train-valid and test splits and get the data loaders
def get_standard_data_loaders(train_batch_size, valid_batch_size = None, test_batch_size = None, transforms = None, svhn_location = './svhn_dataset', train_prop = 0.85) :

	"""
	inputs :

	train_batch_size :
		The batch size for the train split
	valid_batch_size : None
		The batch size for the valid split. None means that the entire data will be returned
	test_batch_size : None
		The batch size for the test split. None means that the entire data will be returned
	transforms : None
		The dictionary of transforms that need to be applied to train/valid/test split. KEYS : 'train', 'valid', 'test'
	svhn_location : './svhn_dataset'
		The place where to save the downloaded dataset
	train_prop : 0.85
		The proportion of the train dataset in train-valid split

	outputs :

	train_loader :
		The pytorch dataset iterator for train split
	valid_loader :
		The pytorch dataset iterator for valid split
	test_loader :
		The pytorch dataset iterator for test split
	"""

	# Define transforms if default
	if transforms is None :
		transforms = {	'train' : torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), 
						'valid' : torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), 
						'test'  : torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
						}

	# Download the datasets
	train_valid_split = torchvision.datasets.SVHN(	os.path.join(svhn_location, 'train'), split = 'train', transform = transforms['train'], 
													target_transform = None, download = True)
	test_split = torchvision.datasets.SVHN(	os.path.join(svhn_location, 'test'), split = 'test', transform = transforms['test'], 
													target_transform = None, download = True)

	# Create the lengths
	train_split_len = int(len(train_valid_split)*train_prop)
	valid_split_len = len(train_valid_split) - train_split_len
	test_split_len = len(test_split)

	# Create the train and valid splits
	train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_split, [train_split_len, valid_split_len])
	test_dataset = test_split

	# Load batch sizes
	if valid_batch_size is None :
		valid_batch_size = valid_split_len
	if test_batch_size is None :
		test_batch_size = test_split_len

	# Create the data loaders
	train_data_loader = torch.utils.data.DataLoader(	train_dataset, batch_size = train_batch_size, 
														shuffle = True, num_workers = 4)
	valid_data_lodaer = torch.utils.data.DataLoader(	valid_dataset, batch_size = valid_batch_size, 
														shuffle = True, num_workers = 4)
	test_data_lodaer = torch.utils.data.DataLoader(		test_dataset, batch_size = test_batch_size, 
														shuffle = True, num_workers = 4)

	return train_data_loader, valid_data_lodaer, test_data_lodaer


if __name__ == '__main__' :
	
	# Get the standard dataset
	train_loader, valid_loader, test_loader = get_standard_data_loaders(	train_batch_size = 16, 
																			valid_batch_size = 16, 
																			test_batch_size = 16)

	# Check repeated iterations
	for i in range(3) :
		print('##################################################')
		for batch_id, (x_train, y_train) in enumerate(train_loader) :
			print('[INFO] Train batch : Index : ', batch_id, ' Shape of x : ', x_train.shape, ' Shape of y : ', y_train.shape)
		print('##################################################')
		for batch_id, (x_valid, y_valid) in enumerate(valid_loader) :
			print('[INFO] Valid batch : Index : ', batch_id, ' Shape of x : ', x_valid.shape, ' Shape of y : ', y_valid.shape)
		print('##################################################')
		for batch_id, (x_test, y_test) in enumerate(test_loader) :
			print('[INFO] Test batch : Index : ', batch_id, ' Shape of x : ', x_test.shape, ' Shape of y : ', y_test.shape)

	# Check shapes and sizes
	print('##################################################')
	for batch_id, (x_train, y_train) in enumerate(train_loader) :
		print('[INFO] Train batch : Index : ', batch_id, ' Shape of x : ', x_train.shape, ' Shape of y : ', y_train.shape)
		print('[INFO] Max value : ', np.max(x_train.cpu().data.numpy()))
		print('[INFO] Min value : ', np.min(x_train.cpu().data.numpy()))
		break
	print('##################################################')
	for batch_id, (x_valid, y_valid) in enumerate(valid_loader) :
		print('[INFO] Valid batch : Index : ', batch_id, ' Shape of x : ', x_valid.shape, ' Shape of y : ', y_valid.shape)
		print('[INFO] Max value : ', np.max(x_valid.cpu().data.numpy()))
		print('[INFO] Min value : ', np.min(x_valid.cpu().data.numpy()))
		break
	print('##################################################')
	for batch_id, (x_test, y_test) in enumerate(test_loader) :
		print('[INFO] Test batch : Index : ', batch_id, ' Shape of x : ', x_test.shape, ' Shape of y : ', y_test.shape)
		print('[INFO] Max value : ', np.max(x_test.cpu().data.numpy()))
		print('[INFO] Min value : ', np.min(x_test.cpu().data.numpy()))
		break

	sys.exit()

	# Quick visualization, if you have these files
	import Print_Updating_Info as print_utils
	import Display_Custom_Plots as display_utils

	print_screen = print_utils.screenToPrintUpdatingInfo()
	display_screen = display_utils.screenToDisplayCustomPlots()

	display_screen.DisplayImagesFromSourcesInCustomPlot(	print_screen = print_screen, 
															image_sources = x_train.cpu().data.numpy().transpose(0, 2, 3, 1), 
															labels_true = y_train.cpu().data.numpy())
	display_screen.DisplayImagesFromSourcesInCustomPlot(	print_screen = print_screen, 
															image_sources = x_valid.cpu().data.numpy().transpose(0, 2, 3, 1), 
															labels_true = y_valid.cpu().data.numpy())
	display_screen.DisplayImagesFromSourcesInCustomPlot(	print_screen = print_screen, 
															image_sources = x_test.cpu().data.numpy().transpose(0, 2, 3, 1), 
															labels_true = y_test.cpu().data.numpy())

	sys.exit()

	# Run sanitization only once
	sanitize_mat("train_32x32.mat")
	sanitize_mat("test_32x32.mat")

	# Split train into train and val. Run this once
	create_train_val_split()

	test_batch_size = 128
	train_batch_size = 128
	valid_batch_size = 128

	test_transform = transforms.Compose([transforms.ToTensor()])
	train_transform = transforms.Compose([transforms.ToTensor()])
	valid_transform = transforms.Compose([transforms.ToTensor()])

	test_dataset = TestDataset(
		mat_file_loc="test_32x32.mat", transform=test_transform)
	train_dataset = TrainDataset(
		mat_file_loc="train_split_32x32.mat", transform=train_transform)
	valid_dataset = ValidDataset(
		mat_file_loc="valid_split_32x32.mat", valid_transform=valid_transform)


	test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
							 shuffle=False, num_workers=4)

	train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
							  shuffle=True, num_workers=4)

	valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size,
							  shuffle=False, num_workers=4)
