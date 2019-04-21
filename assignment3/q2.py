# Dependencies
import torch

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import argparse
import time

from data_loader_utils import BinarizedMNIST
from vae_model import vaeModel

####################################################################################################
# Argument Parser
####################################################################################################
# Define the argument parser
parser = argparse.ArgumentParser('Q2: Variational Auto-Encoder Experiments')

# Numpy, torch and other platform args
parser.add_argument('--seed', type = int, default = 0, help = "The seed to use for the torch stuff")
# Data loader args
parser.add_argument('--batch_size', type = int, default = 20, help = "Batch size for Train, Valid and Test batches")
parser.add_argument('--dataset_path', type = str, default = './', help = 'The path to .npy files of BinarizedMNIST')
# Train/test args
parser.add_argument('--mode', type = str, default = 'train', help = 'Whether to carry out training')
parser.add_argument('--outfolder', type = str, default = './', help = 'Where to store the results')
parser.add_argument('--num_epochs', type = int, default = 20, help = 'The maximum number of training epochs')
parser.add_argument('--model_path', type = str, default = './vae_models', help = 'Where to store the trained model')
parser.add_argument('--model_name', type = str, default = 'EXPT_Q2', help = 'Name of the trained model')
parser.add_argument('--load_path', type = str, default = './vae_models', help = 'Trained model that should be loaded for mode test')
parser.add_argument('--load_path', type = str, default = 'EXPT_Q2', help = 'Name of the trained model to load')

# Parse the arguments
args = parser.parse_args()
seed = args.seed
batch_size = args.batch_size
dataset_path = args.dataset_path
mode = args.mode
num_epochs = args.num_epochs
model_path = args.model_path
model_name = args.model_name
load_path = args.load_path
load_name = args.load_name


####################################################################################################
# Pre-processing
####################################################################################################
# Set seeds
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() :
	torch.cuda.manual_seed(seed)

# Create device
if torch.cuda.is_available() :
	device = 'cuda'
else :
	device = 'cpu'

# Create a folder to store the results of the experiment, without overwriting anything
i = 0
while os.path.exists(os.path.join(outfolder, 'Expt_' + str(i))) :
	i += 1
experiment_folder = os.path.join(outfolder, 'Expt_' + str(i))
# Create the directory
print('[INFO] Creating the directory for storing results at: ' + str(experiment_folder))
os.makedirs(experiment_folder)

# Create a log of all the settings
f_handle = open(os.path.join(experiment_folder, 'Config.txt'), 'w')
f_handle.write('args : ' + str(args) + '\n')
f_handle.write('seed : ' + str(seed) + '\n')
f_handle.write('batch_size : ' + str(batch_size) + '\n')
f_handle.write('dataset_path : ' + str(dataset_path) + '\n')
f_handle.write('mode : ' + str(mode) + '\n')
f_handle.write('device : ' + str(device) + '\n')
f_handle.write('num_epochs : ' + str(num_epochs) + '\n')
f_handle.write('model_path : ' + str(model_path) + '\n')
f_handle.write('model_name : ' + str(model_name) + '\n')
f_handle.write('load_path : ' + str(load_path) + '\n')
f_handle.write('load_name : ' + str(load_name) + '\n')
f_handle.close()


####################################################################################################
# Dataset Instance
####################################################################################################
# Create a data loader
data_loader = BinarizedMNIST(	batch_size = batch_size, dataset_path = dataset_path, 
								is_load_from_npy = True, is_load_from_amat_files = False )


####################################################################################################
# Model Instance
####################################################################################################
# Create the model
vae_model = vaeModel(data_loader = data_loader, device = device, architecture = 'Standard')


####################################################################################################
# Training
####################################################################################################
# If the train mode is given, then train the model
if mode == 'train' :
	vae_model.train(	stopping_criterion = 'Epochs', num_epochs = num_epochs, 
						is_store_early_models = True, model_path = model_path, model_name = model_name, 
						is_write_progress_to_log_file = True, log_file_path = os.path.join(experiment_folder, 'EXPT.log'),
						is_verbose = True)
elif mode == 'test' :
	vae_model.load_model(	model_path = load_path, model_name = load_name)
else :
	print('[ERROR] Wrong mode : ', mode, ' input given.')
	print('[ERROR] Terminating the code ...')
	sys.exit()

####################################################################################################
# Compute ELBO
####################################################################################################
def calculate_elbo(model, split) :

	"""
	inputs :

	model :
		The trained model instance
	split : 
		The data split for which to do this

	outputs :
	
	elbo_list :
		The list of elbo losses
	batch_size_list :
		The list of all the batch sizes
	"""

	elbo_list = []
	batch_size_list = []

	# Reset the split
	model.data_loader.reset_data_split(split = split)

	# While a batch exists ...
	while model.data_loader.is_next_batch_exists(split = split) :
		# Get next batch
		x_batch, y_batch = model.data_loader.get_next_batch(split = split)
		# Add batch size
		batch_size_list.append(int(x_batch.shape[0]))
		# Test using this batch
		_, _, _, _, _, loss = model.test(split = None, x_default = x_batch)
		# Add loss to list
		elbo_list.append(loss)

	return elbo_list, batch_size_list

# Reset the valid data-split
elbo_log_handle = open(os.path.join(experiment_folder, 'ELBO.log'), 'w')
train_elbo_loss_list, train_batch_size_list = calculate_elbo(model = vae_model, split = 'Train')
elbo_log_handle.write(	'[INFO] Train ELBO per sample : ' 
						+ str( float(sum(train_elbo_loss_list))*1.0/(float(sum(train_batch_size_list))*1.0) ) + '\n')
valid_elbo_loss_list, valid_batch_size_list = calculate_elbo(model = vae_model, split = 'Valid')
elbo_log_handle.write(	'[INFO] Valid ELBO per sample : ' 
						+ str( float(sum(valid_elbo_loss_list))*1.0/(float(sum(valid_batch_size_list))*1.0) ) + '\n')
test_elbo_loss_list, test_batch_size_list = calculate_elbo(model = vae_model, split = 'Test')
elbo_log_handle.write(	'[INFO] Test ELBO per sample : ' 
						+ str( float(sum(test_elbo_loss_list))*1.0/(float(sum(test_batch_size_list))*1.0) ) + '\n')


####################################################################################################
# Compute Log-Likelihood
####################################################################################################
# Calculate for the entire dataset the log-likelihood
def calculate_elbo(model, split) :

	"""
	inputs :

	model :
		The trained model instance
	split : 
		The data split for which to do this

	outputs :
	
	ll_list :
		The list of log-likelihood losses
	batch_size_list :
		The list of all the batch sizes
	"""