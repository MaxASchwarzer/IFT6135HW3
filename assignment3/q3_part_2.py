# Dependencies
import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import argparse
import time

from vae_svhn_model import vaeSVHNModel
from svhn_loader import get_standard_data_loaders
from vae_utils import encoderConfig, decoderConfig


####################################################################################################
# Argument Parser
####################################################################################################
# Define the argument parser
parser = argparse.ArgumentParser('Q3: Variational Auto-Encoder Experiments')

# Numpy, torch and other platform args
parser.add_argument('--seed', type = int, default = 0, help = "The seed to use for the torch stuff")
# Data loader args
parser.add_argument('--batch_size', type = int, default = 64, help = "Batch size for Train, Valid and Test batches")
parser.add_argument('--dataset_path', type = str, default = 'Not Needed', help = 'The path to .npy files of BinarizedMNIST')
# Train/test args
parser.add_argument('--mode', type = str, default = 'train', help = 'Whether to carry out training')
parser.add_argument('--outfolder', type = str, default = './', help = 'Where to store the results')
parser.add_argument('--num_epochs', type = int, default = 30, help = 'The maximum number of training epochs')
parser.add_argument('--model_path', type = str, default = './vae_svhn_models', help = 'Where to store the trained model')
parser.add_argument('--model_name', type = str, default = 'EXPT_Q3', help = 'Name of the trained model')
parser.add_argument('--load_path', type = str, default = './vae_svhn_models', help = 'Trained model that should be loaded for mode test')
parser.add_argument('--load_name', type = str, default = 'EXPT_Q3', help = 'Name of the trained model to load')
parser.add_argument('--sample_count', type = int, default = 512, help = 'Number of random samples')

# Parse the arguments
args = parser.parse_args()

seed = args.seed
batch_size = args.batch_size
dataset_path = args.dataset_path
mode = args.mode
outfolder = args.outfolder
num_epochs = args.num_epochs
model_path = args.model_path
model_name = args.model_name
load_path = args.load_path
load_name = args.load_name
sample_count = args.sample_count

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
while os.path.exists(os.path.join(outfolder, 'Expt_SVHN_' + str(i))) :
	i += 1
experiment_folder = os.path.join(outfolder, 'Expt_SVHN_' + str(i))
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
f_handle.write('outfolder : ' + str(outfolder) + '\n')
f_handle.write('device : ' + str(device) + '\n')
f_handle.write('num_epochs : ' + str(num_epochs) + '\n')
f_handle.write('model_path : ' + str(model_path) + '\n')
f_handle.write('model_name : ' + str(model_name) + '\n')
f_handle.write('load_path : ' + str(load_path) + '\n')
f_handle.write('load_name : ' + str(load_name) + '\n')
f_handle.write('sample_count : ' + str(sample_count) + '\n')
f_handle.close()


####################################################################################################
# Dataset Instance
####################################################################################################
# Create the different split data loaders and create a data loader dict
train_data_loader, valid_data_loader, test_data_loader = get_standard_data_loaders(	train_batch_size = batch_size,
																					valid_batch_size = batch_size, 
																					test_batch_size = batch_size)
data_loader_dict = {'train' : train_data_loader, 'valid' : valid_data_loader, 'test' : test_data_loader }


####################################################################################################
# Create encoder and decoder configurations
####################################################################################################
# Parameters
height = 32
width = 32
im_channels = 3
blocks = (1, 1, 1)
channel_dec = 32
channel_enc = 32
zdim = 100

# Create configuration classes
config_enc = encoderConfig(hdim = channel_enc, im_channels = im_channels, blocks = blocks)
config_dec = decoderConfig(	zdim = 100, hdim = channel_dec*2**(len(blocks) - 1), 
							im_channels = im_channels, blocks = blocks, dropout = 0.2)


####################################################################################################
# Create encoder and decoder configurations
####################################################################################################
# Create the model
vae_model = vaeSVHNModel(	data_loaders = data_loader_dict, config_enc = config_enc, 
								config_dec = config_dec, device = device, architecture = 'Standard')
if torch.cuda.is_available() :
	vae_model = vae_model.cuda()


####################################################################################################
# Training
####################################################################################################
# If the train mode is given, then train the model
if mode == 'train' :
	vae_model.train(	stopping_criterion = 'Epochs', num_epochs = num_epochs, 
						is_store_early_models = True, model_path = model_path, model_name = model_name, 
						is_write_progress_to_log_file = True, 
						log_file_path = os.path.join(experiment_folder, 'EXPT_SVHN.log'), 
						is_verbose = True)
elif mode == 'test' :
	vae_model.load_model(	model_path = load_path, model_name = load_name)
else :
	print('[ERROR] Wrong mode : ', mode, ' input given.')
	print('[ERROR] Terminating the code ...')
	sys.exit()


####################################################################################################
# Generate Samples
####################################################################################################
# Create samples
z = np.random.normal(loc = 0.0, scale = 1.0, size = [sample_count, 100])
x_generated = vae_model.generate_data(z_input = z)
np.save(file = os.path.join(experiment_folder, 'z_generation'), arr = z)
np.save(file = os.path.join(experiment_folder, 'x_generation'), arr = x_generated)
# Create a storage directoy
image_dict = os.path.join(experiment_folder, 'VAE_Generated_Images')
os.mkdir(image_dict)
# Save each of the image
x_gen_save = torch.Tensor(x_generated)
for i in range(sample_count) :
	print('[INFO] Saving image : ', i)
	torchvision.utils.save_image(tensor = x_gen_save[i], filename = os.path.join(image_dict, str(i) + str('.jpg')))