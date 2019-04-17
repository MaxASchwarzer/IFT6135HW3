# Dependencies
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

import numpy as np

import sys


# Define a class to hold the convolutional VAE architecture
class VAE(nn.Module) :

	"""
	attributes :

	device :
		The device onto which all the tensors should be moved
	architecture : 'Standard'
		The architecture of the model. SUPPORT : 'Standard'
	latent_size :
		The latent space dimension
	encoder :
		The network modeling q(z|x), the distribution of the latent variable given the data
	decoder :
		The network modeling p(x|z), the distribution of the data given the latent variable
	linear_in :
		The network modeling the linear layer before implementing reparametrization trick
	reparam_mu :
		The network of predicting mu(x) [mean] for given input x, for implementing reparametrization trick
	reparam_log_var :
		The network of predicting log_var(x) [log of variance] for given input x, for implementing reparametrization trick
	linear_out :
		The network modeling the linear layer after implementing reparametrization trick


	methods :

	__init__(self, architecture = 'Standard') :
		The constructor
	encode(self, inputs) :
		A method to compute the encoded version of a batch of images
	reparametrize(self, inputs) :
		A method to compute the reparametrization in terms of the mean and the log-sigma
	sample(self, mean, log_var) :
		A method to compute the samples corresponding to the inputs
	decode(self, inputs) :
		A method to compute the decoding/reconstruction of images from samples
	forward(self, inputs) :
		The forward method for the forward pass
	"""

	# Constructor
	def __init__(self, device, architecture = 'Standard') :

		"""
		inputs :

		device :
			The device onto which all the things should be moved
		architecture : 'Standard'
			The architecture given in the question


		outputs :
		"""

		# Initialize the super class
		super(VAE, self).__init__()

		# Create attributes
		self.device = device
		self.architecture = architecture

		# Create the encoder
		if self.architecture == 'Standard' :	
			# Define the latent size attribute
			self.latent_size = 100
			# Create the sequential encoder model with the given specifications
			self.encoder = nn.Sequential(	nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3)),
											nn.ELU(),
											nn.AvgPool2d(kernel_size = 2, stride = 2),	
											nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3)),
											nn.ELU(),
											nn.AvgPool2d(kernel_size = 2, stride = 2),	
											nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size = (5, 5)),
											nn.ELU()
											)

			# Create the sequential decoder model with the given specifications
			self.decoder = nn.Sequential(	nn.ELU(),
											nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = (5, 5), padding = (4, 4)),
											nn.ELU(),
											nn.UpsamplingBilinear2d(scale_factor = 2),
											nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (3, 3), padding = (2, 2)),
											nn.ELU(),
											nn.UpsamplingBilinear2d(scale_factor = 2),
											nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = (3, 3), padding = (2, 2)),
											nn.ELU(),
											nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = (3, 3), padding = (2, 2)),
											nn.Sigmoid()
											)
			# Create the sequential reparametrizer model with the given specifications
			self.linear_in = nn.Sequential(nn.Linear(in_features = 256, out_features = 100))
			self.reparametrizer_mu = nn.Sequential(nn.Linear(in_features = 100, out_features = 100))
			self.reparametrizer_log_var = nn.Sequential(nn.Linear(in_features = 100, out_features = 100))
			self.linear_out = nn.Sequential(nn.Linear(in_features = 100, out_features = 256))

		else :
			print('[ERROR] Architecture option : ', architecture, ' is NOT implemented.')
			print('[ERROR] Terminating the code ...')
			sys.exit()

			
	# Define a method to compute the encoded version of a batch of images
	def encode(self, inputs) :

		"""
		inputs :

		inputs :	
			Image batch torch tensor. SHAPE : [<batch_size>, <in_channel = 1>, <height>, <width>]

		outputs :

		outputs :
			The encoded torch tensor for the image batch. SHAPE : [<batch_size>, <latent_size = 100>]
		"""

		# BUG FIX! Tensors dropped from GPU are put back
		if torch.cuda.is_available() :
			inputs = inputs.cuda()

		# Get inputs passed through the encoder
		feats = self.encoder(inputs)
		# Flatten the features
		feats_flatten = feats.view(feats.shape[0], -1)
		# Pass the features through the input linear layer before reparametrization
		outputs = self.linear_in(feats_flatten)

		return outputs


	# Define a method to compute the reparametrization in terms of the mean and the log-sigma
	def reparametrize(self, inputs) :

		"""
		inputs :

		inputs :
			The input of encoded images as a torch tensor. SHAPE : [<batch_size>, <latent_size = 100>]

		outputs :

		mean :
			The means corresponding to the inputs as a torch tensor. SHAPE : [<batch_size>, <latent_size = 100>]
		log_var :
			The log variances for the inputs as a torch tensor. SHAPE : [<batch_size>, <latent_size = 100>]
		"""

		# BUG FIX! Tensors dropped from GPU are put back
		if torch.cuda.is_available() :
			inputs = inputs.cuda()

		# Get and return the mean and log-sigma
		mean = self.reparametrizer_mu(inputs)
		log_var = self.reparametrizer_log_var(inputs)

		return mean, log_var


	# Define a method to compute the samples corresponding to the inputs
	def sample(self, mean, log_var) :

		"""
		inputs :

		mean :
			The means corresponding to the inputs as a torch tensor. SHAPE : [<batch_size>, <latent_size = 100>]
		log_var :
			The log variances for the inputs as a torch tensor. SHAPE : [<batch_size>, <latent_size = 100>]

		outputs :

		samples :
			The reparametrization samples from Gaussian(mean, exp(log_var^2)). SHAPE : [<batch_size>, <latent_size = 100>]
		noise :
			The N(0, I) noise corresponding to the samples. SHAPE : [<batch_size>, <latent_size = 100>]
		"""

		# Get N(0, I) samples of appropriate shape
		noise = torch.Tensor(mean.shape).normal_()
		# Set the tensor to device
		noise = noise.to(self.device)
		# Modulate
		std = torch.exp(0.5*log_var)
		samples = mean + torch.mul(noise, std)

		return samples, noise


	# Define a method to compute the decoding/reconstruction of images from samples
	def decode(self, inputs) :

		"""
		inputs :

		inputs :
			Input image batch as torch tensor. SHAPE : [<batch_size>, <channel_in = 1/3>, <height>, <width>]

		outputs :

		reconstr :
			The reconstructed images as torch tensor. SHAPE : [<batch_size>, <channel_out = 1/3>, <height>, <width>]
		"""

		# BUG FIX! Tensors dropped from GPU are put back
		if torch.cuda.is_available() :
			inputs = inputs.cuda()

		# Pass the input through the linear layer
		feats_linear = self.linear_out(inputs)
		# Reshape into a feature
		feats = feats_linear.view(feats_linear.shape[0], 256, 1, -1)
		# Decode the features through the decoder
		reconstr = self.decoder(feats)

		return reconstr


	# Define the forward method for the forward pass
	def forward(self, inputs) :

		"""
		inputs :

		inputs :
			Input image batch as torch tensor. SHAPE : [<batch_size>, <channel_in = 1/3>, <height>, <width>]

		outputs :

		reconstr :
			The reconstructed images as torch tensor. SHAPE : [<batch_size>, <channel_out = 1/3>, <height>, <width>]
		samples :
			The samples for reconstruction as torch tensor. SHAPE : [<batch_size>, <latent_size = 100>]
		noise :
			The noise sampled for the reconstruction as torch tensor. SHAPE : [<batch_size>, <latent_size = 100>]
		mean :
			The means of the latent variables as torch tensor. SHAPE : [<batch_size>, <latent_size = 100>]
		log_var :
			The log-variance of the latent variables as torch tensor. SHAPE : [<batch_size>, <latent_size = 100>]
		"""

		# BUG FIX! Tensors dropped from GPU are put back
		if torch.cuda.is_available() :
			inputs = inputs.cuda()

		# Encode the inputs
		encoded_feats = self.encode(inputs = inputs)
		# Get the means and log-variances
		mean, log_var = self.reparametrize(inputs = encoded_feats)
		# Sample
		samples, noise = self.sample(mean = mean, log_var = log_var)
		# Reconstruct
		reconstr = self.decode(inputs = samples)

		return reconstr, samples, noise, mean, log_var
		

# Pseudo-main
if __name__ == '__main__' :

	vae = VAE(device = torch.device('cpu'))

	# Create a pseudo-batch
	batch_size = 512
	height = 28
	width = 28
	channel_in = 1
	x_batch = torch.Tensor(512, 1, 28, 28).normal_()

	# Get encoding
	x_enc = vae.encode(inputs = x_batch)
	print('[DEBUG] Encoding Shape : ', x_enc.shape) 
	# Reparametrize to get the mean and variance
	x_mean, x_log_var = vae.reparametrize(inputs = x_enc)
	print('[DEBUG] Mean Shape : ', x_mean.shape) 
	print('[DEBUG] Log-Variance Shape : ', x_log_var.shape) 
	# Get samples
	x_samples, x_noise = vae.sample(mean = x_mean, log_var = x_log_var)
	print('[DEBUG] Samples Shape : ', x_samples.shape) 
	print('[DEBUG] Noise Shape : ', x_noise.shape) 
	# Decde the samples
	x_reconstr = vae.decode(inputs = x_samples)
	print('[DEBUG] Reconstructed Images Shape : ', x_reconstr.shape) 

