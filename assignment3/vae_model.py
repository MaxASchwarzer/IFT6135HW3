# Dependencies
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as fn

import numpy as np

import sys
import time
import os

from vae import VAE
from data_loader_utils import BinarizedMNIST


# Define a class to hold the entire VAE as a model, on which training procedure can be defined
class vaeModel(nn.Module) :

	"""
	attributes :

	data_loader :
		A standard data loader instance which has the standard methods
	device :
		The torch.device instance telling the device on which the tensors should be loaded
	architecture : 'Standard'
		The architecture type. SUPPORT : 'Standard'
	optimizer :
		For the training of the architecture, the choice of the optimizer

	methods :

	__init__(self, data_loader, device, architecture = 'Standard')
		The constructor
	set_train_mode(self) :
		A method to set the model to training
	set_eval_mode(self) :
		A method to set the model to testing
	"""

	# Constructor
	def __init__(self, data_loader, device, architecture = 'Standard') :

		"""
		inputs :
		
		data_loader :
			A standard data loader instance which has the standard methods
		device :
			The torch.device instance telling the device on which the tensors should be loaded
		architecture : 'Standard'
			The architecture type. SUPPORT : 'Standard'

		outputs :
		"""

		# Initialize the super class
		super(vaeModel, self).__init__()

		# Create attributes
		self.data_loader = data_loader
		self.device = device
		self.architecture = architecture

		# Create a VAE instance
		self.model = VAE(device = self.device, architecture = self.architecture)

		# Create the optimizer
		if self.architecture == 'Standard' :
			# Add ADAM optimizer, with learning rate of 3x10^{-4}
			self.optimizer = optim.Adam(params = self.model.parameters(), lr = 3e-4)

		else :
			print('[ERROR] Architecture option : ', architecture, ' is NOT implemented.')
			print('[ERROR] Terminating the code ...')
			sys.exit()


	# Define a method to set the model to training
	def set_train_mode(self) :

		"""
		inputs :

		outputs :
		"""

		self.model.train()


	# Define a method to set the model to testing
	def set_eval_mode(self) :

		"""
		inputs :

		outputs :
		"""

		self.model.eval()


	# Define a method to compute the loss of the architecture
	def evaluate_loss(self, x, x_reconstr, mean, log_var) :

		"""
		inputs :

		x :
			The input image batch as torch tensor. SHAPE : [<batch_size>, <channel_in = 1/3>, <width>, <height>]
		x_reconstr :
			The reconstructed image batch as torch tensor. SHAPE : [<batch_size>, <channel_in = 1/3>, <width>, <height>]
		mean :
			The mean of the image batch. SHAPE : [<batch_size>, <latent_size = 100>]
		log_var :
			The log-variance of the image batch. SHAPE : [<batch_size>, <latent_size = 100>]

		outputs :

		loss :
			The net LOSS (negative ELBO) corresponding to the batch
		"""

		# Compute binary cross-entropy
		binary_cross_entropy_loss = fn.binary_cross_entropy(input = x_reconstr.view(-1, 784), target = x.view(-1, 784), reduction = 'sum') 
		# print('[DEBUG] BCE Loss Shape : ', binary_cross_entropy_loss.shape)
		# Compute KL-divergence
		kl_loss = -0.5*torch.sum(1.0 + log_var - mean.pow(2) - log_var.exp())
		# print('[DEBUG] KL-Divergence Loss Shape : ', kl_loss.shape)
		# Net loss
		loss = binary_cross_entropy_loss + kl_loss

		return loss


	# Define a method to save the model
	def save_model(self, model_path = './Models', model_name = 'EXPT') :

		"""
		inputs :

		model_path : './Models'
			The path to which the model needs to be saved
		model_name : 'EXPT'
			The name of the experiment
		"""

		"""
		outputs :
		"""

		# Save the weights to the path
		save_path = os.path.join(model_path, model_name)
		torch.save(self.state_dict(), save_path)


	# Define a function to load the model from a path
	def load_model(self, model_path = './vae_models', model_name = 'EXPT') :

		"""
		inputs :

		model_path : './Models'
			The path to which the model needs to be saved
		model_name : 'EXPT'
			The name of the experiment
		"""

		"""
		outputs :
		"""

		load_path = os.path.join(model_path, model_name)
		self.load_state_dict(torch.load(load_path))


	# Define a method to train the model 
	def train(self, stopping_criterion = 'Epochs', num_epochs = 5000, is_store_early_models = True, model_path = './vae_models', model_name = 'EXPT', is_write_progress_to_log_file = True, log_file_path = 'EXPT.log', is_verbose = True) :

		"""
		inputs :

		stopping_criterion : 'Epochs'
			The stopping criterion to be used to cut training. SUPPORT : 'Epochs'
		num_epochs : 5000
			The number of epochs to be used for training, in case the stopping_criterion is 'Epochs'
		is_store_early_models : True
			Whether to store the model based on improvements in the validation phase
		model_path : './Models'
			The path to which the model needs to be saved
		model_name : 'EXPT'
			The name of the experiment
		is_write_progress_to_log_file : True
			Whether to write the progress to a log file
		log_file_path : 'EXPT.log'
			The path to the log-file where progress needs to be written
		is_verbose : True
			Whether to display the information

		outputs :
		"""

		is_continue_training = True
		best_val_loss = None
		tr_loss = 0.0
		val_loss = 0.0
		te_loss = 0.0

		# If we want to store the progress, create a file
		if is_write_progress_to_log_file :
			f_log = open(log_file_path, 'w')

		# Set the stopping criterion
		if stopping_criterion == 'Epochs' :
			threshold_stop_training = num_epochs
			trigger_stop_training = 0

		# Set the model mode to training
		self.set_train_mode()

		# While to continue training ...
		while is_continue_training :

			iteration_count = 0
			# For each epoch, reset the train split
			self.data_loader.reset_data_split(split = 'Train')

			# While there is a new training batch
			while self.data_loader.is_next_batch_exists(split = 'Train') :

				# Start the timer
				time_start = time.time()

				# Update iteration counter
				iteration_count += 1

				# Load a batch
				x_batch, y_batch = self.data_loader.get_next_batch(split = 'Train')
				# Reshape to image
				x_batch = np.reshape(x_batch, [-1, 28, 28, 1])
				# Convert into a tensor
				x_tensor = torch.Tensor(x_batch)
				# Make first dimension as channel
				x_tensor = x_tensor.permute(0, 3, 1, 2)
				# Load to device
				if 'cuda' in self.device :
					x = x_tensor.cuda()

				# Set the optimizer gradient to 0
				self.optimizer.zero_grad()

				# Get the samples, noise, mean and log-variances from the forward pass of the VAE
				reconstr, samples, noise, mean, log_var = self.model(inputs = x)
				# Compute the loss
				loss = self.evaluate_loss(x = x, x_reconstr = reconstr, mean = mean, log_var = log_var)
				# Back-propagate the gradients
				loss.backward()
				# Update the weights
				self.optimizer.step()

				# Get the loss as numpy value
				tr_loss = loss.cpu().data.numpy()
				# Normalize per sample
				tr_loss = (tr_loss*1.0)/(1.0*x.shape[0])

				# Display
				if is_verbose :
					print('[INFO] Epoch : ', trigger_stop_training, ' Iteration : ', iteration_count, ' Training Loss : ', tr_loss)

				# End the timer
				time_stop = time.time()
				# Print the stats of time
				print('[INFO] Time per iteration : ', time_stop - time_start)

			# Evaluate the model
			val_loss = self.test(split = 'Validation')
			te_loss = self.test(split = 'Test')
			if is_verbose :
				print('[INFO] Epoch : ', trigger_stop_training, ' Iteration : ', iteration_count, ' Validation Loss : ', val_loss)
				print('[INFO] Epoch : ', trigger_stop_training, ' Iteration : ', iteration_count, ' Testing Loss : ', te_loss)

			if is_write_progress_to_log_file :
				f_log.write('Epoch : ' + str(trigger_stop_training) + ' TrainingLoss : ' + str(tr_loss) + ' ValidationLoss : ' + str(val_loss) + ' TestingLoss : ' + str(te_loss) + '\n')

			# If the validation loss has improved, save
			if best_val_loss is None :
				best_val_loss = val_loss
			if val_loss < best_val_loss :
				if is_store_early_models :
					if is_verbose :
						print('[INFO] Storing Best Model with Previous Validation Loss : ', best_val_loss, ' and Improved Validation Loss : ', val_loss)
					if is_write_progress_to_log_file :
						f_log.write('Epoch : ' + str(trigger_stop_training) + 'Storing Best Model with Previous Validation Loss : ' + str(best_val_loss) + ' and Improved Validation Loss : ' + str(val_loss) + '\n')
					self.save_model(model_path = model_path, model_name = model_name)
					# Reset the best validation loss to the new value
				best_val_loss = val_loss

			# Re-evaluate the criterion for stopping
			if stopping_criterion == 'Epochs' :
				trigger_stop_training += 1
				if trigger_stop_training >= threshold_stop_training :
					is_continue_training = False
				else :
					is_continue_training = True

		# Close the file
		if is_write_progress_to_log_file :
			f_log.close()


	# Define a method to test the model
	def test(self, split) :

		"""
		inputs :

		split :
			The dataset split on which we want to test the model

		outputs :

		reconstr_np :
			The reconstruction corresponding to the split
		samples_np :
			The numpy array of samples
		noise_np :
			The numpy array of noise
		mean_np :
			The numpy array of mean
		log_var_np :
			The numpy array of log-variances
		loss_np :
			The loss corresponding to the split
		"""

		# Set the mode to testing
		self.set_eval_mode()

		# Load the data split
		x_batch, y_batch = self.data_loader.get_data_split(split = split)
		# Reshape to image
		x_batch = np.reshape(x_batch, [-1, 28, 28, 1])
		# Convert into a tensor
		x_tensor = torch.Tensor(x_batch)
		# Make first dimension as channel
		x_tensor = x_tensor.permute(0, 3, 1, 2)
		# Load to device
		if 'cuda' in self.device :
			x = x_tensor.cuda()

		# Set the optimizer gradient to 0
		self.optimizer.zero_grad()

		# Get the samples, noise, mean and log-variances from the forward pass of the VAE
		reconstr, samples, noise, mean, log_var = self.model(inputs = x)
		# Compute the loss
		loss = self.evaluate_loss(x = x, x_reconstr = reconstr, mean = mean, log_var = log_var)

		# Get the numpy arrays corresponding to everything
		reconstr_np = reconstr.cpu().data.numpy()
		samples_np = samples.cpu().data.numpy()
		noise_np = nosie.cpu().data.numpy()
		mean_np = mean.cpu().data.numpy()
		log_var_np = log_var.cpu().data.numpy()
		loss_np = loss.cpu().data.numpy()
		# Normalize the loss
		loss_np = (loss_np*1.0)/(1.0*x.shape[0])

		return reconstr_np, samples_np, noise_np, mean_np, log_var_np, loss_np


# Pseudo-main
if __name__ == '__main__' :

	# Create a dataset instance
	data_loader = BinarizedMNIST(batch_size = 1024)

	# Create a model
	if torch.cuda.is_available() :
		device = 'cuda'
	else :
		device = 'cpu'
	vae_model = vaeModel(data_loader = data_loader, device = device)
	if 'cuda' in device :
		vae_model = vae_model.cuda()

	# Train the model
	vae_model.train(num_epochs = 1)