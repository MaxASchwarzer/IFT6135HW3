# Dependencies
from __future__ import division, print_function

import torch as T
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np

import sys
import time
import os


# Define a class to hold the encoder configuration utilities
class encoderConfig(object) :

	"""
	attributes :

	hdim :
		The initial value of hidden size. Check encoder (Discriminator-like) architecture for usage
	im_channels :
		The input image channels. Should be 3 for SVHN dataset. 
	blocks :
		The list containing the block structure

	methods :

	__init__(self, hdim, im_channels, blocks) :
		The constructor
	"""
	
	# Define the constructor
	def __init__(self, hdim, im_channels, blocks) :

		"""
		inputs :

		hdim :
			The initial value of hidden size. Check encoder (Discriminator-like) architecture for usage
		im_channels :
			The input image channels. Should be 3 for SVHN dataset. 
		blocks :
			The list containing the block structure

		outputs :
		"""

		self.hdim = hdim
		self.im_channels = im_channels
		self.blocks = blocks


# Define a class to hold the decoder configuration utilities
class decoderConfig(object) :

	"""
	attributes :

	zdim :
		The latent representation dimension. Should be 100
	hdim :
		The initial value for the hidden dimension
	im_channels :
		The number of input data channels. Should be 3
	blocks :
		The list of block structure
	dropout :
		The drop-out rate for neurons

	methods :

	__init__(self, zdim, hdim, im_channels, blocks, dropout) :
		The constructor
	"""

	def __init__(self, zdim, hdim, im_channels, blocks, dropout) :

		"""
		inputs :

		zdim :
			The latent representation dimension. Should be 100
		hdim :
			The initial value for the hidden dimension
		im_channels :
			The number of input data channels. Should be 3
		blocks :
			The list of block structure
		dropout :
			The drop-out rate for neurons


		outputs :
		"""

		self.zdim = zdim
		self.hdim = hdim
		self.im_channels = im_channels
		self.blocks = blocks
		self.dropout = dropout


# Define a class to hold the resnet block
class ResBlock(nn.Module):

	"""
	attributes :

	rescaler :
		A block that either performs 2D conv or implements spectral norm
	rescale :
		Whether rescaling was performed
	bn :
		Whether a normalization is applied
	stride :
		The stride for the layers
	conv1 :
		The first convolutional layer
	dropout1 :
		The fiest dropout layer
	bn0 :
		The 0-th (input's) batch normalization
	bn1 :
		The 1-st batch normalization
	bn2 :
		The 2-nd batch normalization
	conv2 :
		The second convolutional layer for the block
	dropout2 :
		The second dropout layer
	relu1 :
		The first ReLU layer
	relu2 :
		The second ReLU layer

	methods :

	__init__(self, inchannels, outchannels, stride=2, dropout=0.5, bn=False, ln=False, sn=False, h=None) :
		The constructor
	forward(self, x) :
		A method to implement the forward pass 
	"""

	def __init__(self, inchannels, outchannels, stride = 2, dropout = 0.5, bn = False, ln = False, sn = False, h = None):
		super(ResBlock, self).__init__()
		if inchannels != outchannels:
			if stride > 0:
				self.rescaler = nn.Sequential(
									nn.Conv2d(inchannels, outchannels, 3, padding=1, stride=stride),
									nn.LeakyReLU(0.2)
									)
				if sn:
					self.rescaler = nn.utils.spectral_norm(self.rescaler)
			elif stride < 0:
				self.rescaler = nn.Sequential(
									nn.Conv2d(inchannels, outchannels, 3, padding=1, stride=1),
									nn.LeakyReLU(0.2),
									)
			self.rescale = True
		else:
			self.rescale = False

		self.bn = bn or ln

		self.stride = stride
		self.conv1 = (nn.Conv2d(outchannels, outchannels, 3, padding=1))
		self.dropout1 = nn.Dropout(dropout)
		if bn:
			self.bn0 = nn.BatchNorm2d(outchannels)
			self.bn1 = nn.BatchNorm2d(outchannels)
			self.bn2 = nn.BatchNorm2d(outchannels)
		elif ln:
			self.bn0 = nn.LayerNorm((outchannels, h, h))
			self.bn1 = nn.LayerNorm((outchannels, h, h))
			self.bn2 = nn.LayerNorm((outchannels, h, h))
		self.conv2 = (nn.Conv2d(outchannels, outchannels, 3, padding=1))
		self.dropout2 = nn.Dropout(dropout)
		if sn:
			self.conv1 = nn.utils.spectral_norm(self.conv1)
			self.conv2 = nn.utils.spectral_norm(self.conv2)

		self.relu1 = nn.LeakyReLU(0.2)
		self.relu2 = nn.LeakyReLU(0.2)

	def forward(self, x):
		if self.rescale:
			x = self.rescaler(x)

			# Can't upsample inside the sequential, sadly
			if self.stride < 0:
				x = F.interpolate(x, scale_factor=-self.stride, mode="bilinear", align_corners=True)

			if self.bn:
				x = self.bn0(x)
		x1 = self.conv1(x)
		x1 = self.relu1(x1)
		if self.bn:
			x1 = self.bn1(x1)
		x1 = self.dropout1(x1)
		x2 = self.conv2(x1)
		x2 = self.relu2(x2)
		if self.bn:
			x2 = self.bn2(x2)
		x2 = self.dropout2(x2)

		output = x2 + x

		# print('[DEBUG][forward:ResBlock] Forward output shape : ', output.shape)
		return output


class Generator(nn.Module):
	def __init__(self, zdim, hdim, im_channels=3, blocks=[2, 2, 2, 2], dropout=0.2):
		super(Generator, self).__init__()

		self.initial = nn.Linear(zdim, 16*hdim)

		self.blocks = nn.ModuleList()

		current_dim = hdim
		new_dim = int(hdim/2)
		new_dim = hdim//2
		h = 8
		for nblocks in blocks:
			for block in range(nblocks):
				if new_dim != current_dim:
					self.blocks.append(ResBlock(current_dim, new_dim, stride=-2, dropout=dropout, bn=True, sn=False, h=h))
					current_dim = new_dim
				self.blocks.append(ResBlock(current_dim, current_dim, stride=1, dropout=dropout, bn=True, sn=False, h=h))
			# new_dim = int(current_dim/2)
			new_dim = current_dim//2
			h *= 2

		self.final = (nn.Conv2d(current_dim, im_channels, 7, padding=3))
		self.final_bn = nn.BatchNorm2d(im_channels)

	def forward(self, x):
		current = self.initial(x)
		current = current.view(x.shape[0], -1, 4, 4)

		for block in self.blocks:
			current = block(current)

		output = T.sigmoid(self.final(current))
		# print('[DEBUG][forward:Generator] Forward output shape : ', output.shape)
		return output


class ConvDiscriminator(nn.Module):
	def __init__(self, hdim, im_channels=3, blocks=[2, 2, 2, 2], dropout=0):
		super(ConvDiscriminator, self).__init__()
		self.blocks = nn.ModuleList()

		self.initial = nn.Conv2d(im_channels, hdim, 7, padding=3)
		self.relu = nn.LeakyReLU(0.2)

		h = 16
		current_dim = hdim
		new_dim = current_dim*2
		for nblocks in blocks:
			for block in range(nblocks):
				if new_dim != current_dim:
					self.blocks.append(ResBlock(current_dim, new_dim, stride=2, dropout=dropout, ln=True, sn=False, h=h))
					current_dim = new_dim
				self.blocks.append(ResBlock(current_dim, current_dim, stride=1, dropout=dropout, ln=True, sn=False, h=h))
			new_dim = current_dim*2
			# h = int(h/2)
			h = h//2

		self.fc1 = (nn.Linear(current_dim*16, current_dim))
		self.ln1 = nn.LayerNorm(current_dim, elementwise_affine=False)
		self.fc2 = (nn.Linear(current_dim, 100))

	def forward(self, x):
		current = self.initial(x)
		current = self.relu(current)

		for block in self.blocks:
			current = block(current)

		# print('[DEBUG][forward:ConvDiscriminator] Current shape : ', current.shape)		
		current = self.fc1(current.flatten(1, -1))
		current = self.ln1(current)
		current = self.relu(current)
		output = self.fc2(current)

		# print('[DEBUG][forward:ConvDiscriminator] Forward output shape : ', output.shape)

		return output


# Pseudo-main
if __name__ == '__main__' :

	pass