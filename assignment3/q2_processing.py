# Dependencies
import torch
import torchvision

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
parser.add_argument('--batch_size', type = int, default = 64, help = "Batch size for Train, Valid and Test batches")
parser.add_argument('--dataset_path', type = str, default = './', help = 'The path to .npy files of BinarizedMNIST')
# Train/test args
parser.add_argument('--mode', type = str, default = 'train', help = 'Whether to carry out training')
parser.add_argument('--outfolder', type = str, default = './', help = 'Where to store the results')
parser.add_argument('--num_epochs', type = int, default = 20, help = 'The maximum number of training epochs')
parser.add_argument('--model_path', type = str, default = './vae_models', help = 'Where to store the trained model')
parser.add_argument('--model_name', type = str, default = 'EXPT_Q2', help = 'Name of the trained model')
parser.add_argument('--load_path', type = str, default = './vae_models', help = 'Trained model that should be loaded for mode test')
parser.add_argument('--load_name', type = str, default = 'EXPT_Q2', help = 'Name of the trained model to load')
parser.add_argument('--sample_count', type = int, default = 1000, help = 'Number of random samples')
parser.add_argument('--display_im_count', type = int, default = 200, help = 'Number of random samples to display')
parser.add_argument('--display_im_count_per_row', type = int, default = 10, help = 'Number of random samples to display per row')

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
display_im_count = args.display_im_count
display_im_count_per_row = args.display_im_count_per_row


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
f_handle.write('outfolder : ' + str(outfolder) + '\n')
f_handle.write('device : ' + str(device) + '\n')
f_handle.write('num_epochs : ' + str(num_epochs) + '\n')
f_handle.write('model_path : ' + str(model_path) + '\n')
f_handle.write('model_name : ' + str(model_name) + '\n')
f_handle.write('load_path : ' + str(load_path) + '\n')
f_handle.write('load_name : ' + str(load_name) + '\n')
f_handle.write('sample_count : ' + str(sample_count) + '\n')
f_handle.write('display_im_count : ' + str(display_im_count) + '\n')
f_handle.write('display_im_count_per_row : ' + str(display_im_count_per_row) + '\n')
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
if torch.cuda.is_available() :
	vae_model = vae_model.cuda()


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
# Generate Samples
####################################################################################################
# Create samples
z = np.random.normal(loc = 0.0, scale = 1.0, size = [sample_count, 100])
x_generated = vae_model.generate_data(z_input = z)
np.save(file = os.path.join(experiment_folder, 'z_generation'), arr = z)
np.save(file = os.path.join(experiment_folder, 'x_generation'), arr = x_generated)

x_gen_torch = torch.Tensor(x_generated)
# Select randomly the number of images to display
indices = np.random.randint(low = 0, high = sample_count, size = [display_im_count, ]).astype(np.int32)
x_to_show = x_gen_torch[indices]
# Put to a figure
torchvision.utils.save_image(tensor = x_to_show, filename = os.path.join(experiment_folder, 'Generated_Samples' + str('.jpg')), nrow = display_im_count_per_row)