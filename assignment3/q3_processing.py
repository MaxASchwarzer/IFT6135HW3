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
parser.add_argument('--repeat_q_3_1', type = int, default = 10, help = 'Number of times to repeat Q.3.2')
parser.add_argument('--repeat_q_3_2', type = int, default = 10, help = 'Number of times to repeat Q.3.3')
parser.add_argument('--perturb', type = float, default = 1e-1, help = 'Per dimension perturbation')
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
repeat_q_3_1 = args.repeat_q_3_1
repeat_q_3_2 = args.repeat_q_3_2
perturb = args.perturb
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
f_handle.write('repeat_q_3_1 : ' + str(repeat_q_3_1) + '\n')
f_handle.write('repeat_q_3_2 : ' + str(repeat_q_3_2) + '\n')
f_handle.write('perturb : ' + str(perturb) + '\n')
f_handle.write('display_im_count : ' + str(display_im_count) + '\n')
f_handle.write('display_im_count_per_row : ' + str(display_im_count_per_row) + '\n')
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
# Q.3.1 Generate Samples
####################################################################################################
# Create samples
z = np.random.normal(loc = 0.0, scale = 1.0, size = [sample_count, 100])
x_generated = vae_model.generate_data(z_input = z)
np.save(file = os.path.join(experiment_folder, 'z_generation'), arr = z)
np.save(file = os.path.join(experiment_folder, 'x_generation'), arr = x_generated)
# Create a storage directoy
image_dict = os.path.join(experiment_folder, 'Q_3_1')
os.mkdir(image_dict)
# Save each of the image
x_gen_save = torch.Tensor(x_generated)
for i in range(sample_count) :
	print('[INFO] Saving image : ', i)
	torchvision.utils.save_image(tensor = x_gen_save[i], filename = os.path.join(image_dict, str(i) + str('.jpg')))
# Save an answer image
x_gen_torch = torch.Tensor(x_generated)
# Select randomly the number of images to display
indices = np.random.randint(low = 0, high = sample_count, size = [display_im_count, ]).astype(np.int32)
x_to_show = x_gen_torch[indices]
# Put to a figure
torchvision.utils.save_image(tensor = x_to_show, filename = os.path.join(experiment_folder, 'Generated_Samples' + str('.jpg')), nrow = display_im_count_per_row)


####################################################################################################
# Q.3.2 Noise-and-Dimension
####################################################################################################
# Repeat the experiment
q31_dict = os.path.join(experiment_folder, 'Q_3_2')
os.mkdir(q31_dict)
# Repeat ...
for a_run in range(repeat_q_3_1) :

	print('[INFO] Run of Q.3.2 : ', a_run)

	run_dict = os.path.join(q31_dict, str(a_run))
	os.mkdir(run_dict)
	# Get a noise
	z = np.random.normal(loc = 0.0, scale = 1.0, size = [1, 100])	
	np.save(file = os.path.join(run_dict, 'z_generation'), arr = z)
	x_generated = vae_model.generate_data(z_input = z)
	x_gen_torch = torch.Tensor(x_generated)
	torchvision.utils.save_image(tensor = x_gen_torch, filename = os.path.join(run_dict, str('Base.jpg')))
	# Perturbations
	perturb_np = perturb*np.eye(100)
	# For each perturbation ...
	for i in range(100) :
		z_i = z + perturb_np[i].reshape([1, 100])
		x_i = vae_model.generate_data(z_input = z_i)
		x_i_torch = torch.Tensor(x_i)
		torchvision.utils.save_image(tensor = x_i_torch, filename = os.path.join(run_dict, str(i) + str('.jpg')))


####################################################################################################
# Q.3.3 Interpolation in Latent and Data-Spaces
####################################################################################################
# Repeat the experiment
q32_dict = os.path.join(experiment_folder, 'Q_3_3')
os.mkdir(q32_dict)
# Repeat ...
for a_run in range(repeat_q_3_2) :

	print('[INFO] Run of Q.3.3 : ', a_run)

	run_dict = os.path.join(q32_dict, str(a_run))
	os.mkdir(run_dict)
	# Get two random noises
	z_1 = np.random.normal(loc = 0.0, scale = 1.0, size = [1, 100])	
	z_2 = np.random.normal(loc = 0.0, scale = 1.0, size = [1, 100])	
	np.save(file = os.path.join(run_dict, 'z_1_generation'), arr = z_1)
	np.save(file = os.path.join(run_dict, 'z_2_generation'), arr = z_2)
	# Create the linspace of all the intermediate alphas
	alphas = np.linspace(start = 0.0, stop = 1.0, num = 11)
	# For each alpha ...
	for i in range(alphas.shape[0]) :
		# Get the interpolated noise
		z_i = alphas[i] * z_1 + (1.0 - alphas[i]) * z_2
		x_i = vae_model.generate_data(z_input = z_i)
		x_i_torch = torch.Tensor(x_i)
		torchvision.utils.save_image(tensor = x_i_torch, filename = os.path.join(run_dict, str(i) + str('_generated.jpg')))
	# Now, get the images for the z_1, z_2
	x_1 = vae_model.generate_data(z_input = z_1)
	x_2 = vae_model.generate_data(z_input = z_2)
	# Create a list of all interpolated images
	for i in range(alphas.shape[0]) :
		# Get interpolated images
		x_i = alphas[i] * x_1 + (1.0 - alphas[i]) * x_2
		x_i_torch = torch.Tensor(x_i)
		torchvision.utils.save_image(tensor = x_i_torch, filename = os.path.join(run_dict, str(i) + str('_interpolated.jpg')))