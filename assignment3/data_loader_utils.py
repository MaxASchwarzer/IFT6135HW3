# Dependencies
import numpy as np

import os
import sys


# Define a function to convert the AMAT files of the dataset into .npy files and store
def amat_to_npy(dataset_path = './') :

	"""
	inputs :

	dataset_path : './'
		The path at which 'binarized_mnist_<train,valid,test>.amat' files are present

	outputs :
	"""

	# Create the train, valid, test file paths
	path_train = os.path.join(dataset_path, 'binarized_mnist_train.amat')
	path_valid = os.path.join(dataset_path, 'binarized_mnist_valid.amat')
	path_test = os.path.join(dataset_path, 'binarized_mnist_test.amat')

	# Load the files
	f_train = open(path_train, 'r')	
	f_valid = open(path_valid, 'r')
	f_test = open(path_test, 'r')

	# Read lines from the files
	lines_train = f_train.readlines()
	lines_valid = f_valid.readlines()
	lines_test = f_test.readlines()	

	# Get the train, valid and test np arrays
	x_train = np.array([[int(a_digit) for a_digit in a_line.split()] for a_line in lines_train]).astype(np.float32)
	y_train = np.array([[int(a_digit) for a_digit in a_line.split()] for a_line in lines_train]).astype(np.float32)
	x_valid = np.array([[int(a_digit) for a_digit in a_line.split()] for a_line in lines_valid]).astype(np.float32)
	y_valid = np.array([[int(a_digit) for a_digit in a_line.split()] for a_line in lines_valid]).astype(np.float32)
	x_test = np.array([[int(a_digit) for a_digit in a_line.split()] for a_line in lines_test]).astype(np.float32)
	y_test = np.array([[int(a_digit) for a_digit in a_line.split()] for a_line in lines_test]).astype(np.float32)
	# Create dictionaries to be stored
	dict_train = {'x' : x_train, 'y' : y_train}
	dict_valid = {'x' : x_valid, 'y' : y_valid}
	dict_test = {'x' : x_test, 'y' : y_test}

	# Create the paths to the .npy datasets
	save_train = os.path.join(dataset_path, 'binarized_mnist_train.npy')
	save_valid = os.path.join(dataset_path, 'binarized_mnist_valid.npy')
	save_test = os.path.join(dataset_path, 'binarized_mnist_test.npy')

	# Save at the respective dicts at the respective paths
	np.save(save_train, dict_train)	
	np.save(save_valid, dict_valid)
	np.save(save_test, dict_test)


# Define a function to create/overwrite the .npy files for the dataset
def overwrite_npy(dataset_path = './') :

	"""
	inputs :

	dataset_path : './'
		The path at which 'binarized_mnist_<train,valid,test>.amat' files are present

	outputs :
	"""

	# Get the .npy paths
	save_train = os.path.join(dataset_path, 'binarized_mnist_train.npy')
	save_valid = os.path.join(dataset_path, 'binarized_mnist_valid.npy')
	save_test = os.path.join(dataset_path, 'binarized_mnist_test.npy')

	# Remove current .npy, if any
	os.system('rm -rf ' + save_train)
	os.system('rm -rf ' + save_valid)
	os.system('rm -rf ' + save_test)

	# Re-create the .npy
	amat_to_npy(dataset_path = dataset_path)

####################################################################################################


# Define a class to hold the data loader for MNIST
class BinarizedMNIST(object) :

	# Constructor
	def __init__(self, batch_size = 256, dataset_path = './', is_load_from_npy = True, is_load_from_amat_files = False) :

		"""
		inputs :

		batch_size : 256
			The number of elements in one batch
		dataset_path : './'
			The path where the dataset files reside. These can be .amat files or the .npy files reside
		is_load_from_npy : True
			Whether to load the dataset from the .npy files
		is_load_from_amat_files : False
			Whether to load the dataset raw from the .amat files. We do this to avoid pickle issues in python2 and python3 (of which, there are MANY!)
		"""

		# If loading is from .amat ...
		if is_load_from_amat_files :

			save_train = os.path.join(dataset_path, 'binarized_mnist_train.amat')
			save_valid = os.path.join(dataset_path, 'binarized_mnist_valid.amat')
			save_test = os.path.join(dataset_path, 'binarized_mnist_test.amat')
			
			# Load the .amat files for the binarized version
			f_train = open(save_train, 'r')
			f_valid = open(save_valid, 'r')
			f_test = open(save_test, 'r')

			# Create the (randomized training), validation and testing splits (no need to randomize these 2)
			lines_train = f_train.readlines()
			lines_valid = f_valid.readlines()
			lines_test = f_test.readlines()
			
			self.x_train = np.array([[int(a_digit) for a_digit in a_line.split()] for a_line in lines_train]).astype(np.float32)
			self.y_train = np.array([[int(a_digit) for a_digit in a_line.split()] for a_line in lines_train]).astype(np.float32)
			self.x_valid = np.array([[int(a_digit) for a_digit in a_line.split()] for a_line in lines_valid]).astype(np.float32)
			self.y_valid = np.array([[int(a_digit) for a_digit in a_line.split()] for a_line in lines_valid]).astype(np.float32)
			self.x_test = np.array([[int(a_digit) for a_digit in a_line.split()] for a_line in lines_test]).astype(np.float32)
			self.y_test = np.array([[int(a_digit) for a_digit in a_line.split()] for a_line in lines_test]).astype(np.float32)

		# If loading is from .npy ...
		elif is_load_from_npy :

			# Create paths for loading
			save_train = os.path.join(dataset_path, 'binarized_mnist_train.npy')
			save_valid = os.path.join(dataset_path, 'binarized_mnist_valid.npy')
			save_test = os.path.join(dataset_path, 'binarized_mnist_test.npy')

			# Load the .npy files and extract the dictionaries
			train_dataset_obj = np.load(save_train)
			train_dataset = train_dataset_obj.item()
			valid_dataset_obj = np.load(save_valid)
			valid_dataset = valid_dataset_obj.item()
			test_dataset_obj = np.load(save_test)
			test_dataset = test_dataset_obj.item()

			# Create the splits
			self.x_train = train_dataset['x']			
			self.y_train = train_dataset['y']
			self.x_valid = valid_dataset['x']
			self.y_valid = valid_dataset['y']
			self.x_test = test_dataset['x']
			self.y_test = test_dataset['y']

		# This place should never be reached by any code ...
		else :
			print('[ERROR] Unimplemented option to load directly.')
			print('[ERROR] Terminating the code ...')
			sys.exit(status)

		# Create a random interation
		self.iteration_train = np.random.permutation(self.x_train.shape[0])
		self.x_train = self.x_train[self.iteration_train]
		self.y_train = self.y_train[self.iteration_train]

		# Create a pointer to current batch start point
		self.current_batch_start_train = 0
		self.current_batch_start_valid = 0
		self.current_batch_start_test = 0

		# Store attribute for batch size
		self.batch_size = batch_size


	# Define a method to check if the set has next batch
	def is_next_batch_exists(self, split = 'Train') :

		"""
		inputs :

		split : 'Train'
			The split in which we want to check if the next batch exists
		"""

		"""
		outputs :

		is_exists (implicit) :
			Whether there is a next batch in the set
		"""

		# Just check if the current batch start is smaller than the length of split
		if split == 'Train' :
			return self.current_batch_start_train < self.x_train.shape[0]
		elif split == 'Valid' :
			return self.current_batch_start_valid < self.x_valid.shape[0]
		elif split == 'Test' :
			return self.current_batch_start_test < self.x_test.shape[0]
		
		# This place should never be reached by any code ...
		else :
			print('[ERROR] Wrong split is querried : ', str(split))
			print('[ERROR] Terminating the code ...')
			sys.exit(status)


	# Define a method to get the next batch from the split
	def get_next_batch(self, split = 'Train') :

		"""
		inputs :

		split : 'Train'
			The split in which we want to check if the next batch exists
		"""

		"""
		outputs :

		x_batch :
			The batch of the data. SHAPE : [<batch_size>, <feat_dim>]
		y_batch ;
			The batch of the labels. SHAPE : [<batch_size>, ]
		"""

		# Return the batch and increment the counter
		if split == 'Train' :
			start_point = self.current_batch_start_train
			end_point = np.minimum(start_point + self.batch_size, self.x_train.shape[0])
			x_batch = self.x_train[start_point : end_point]
			y_batch = self.y_train[start_point : end_point]
			self.current_batch_start_train = end_point
		elif split == 'Valid' :
			start_point = self.current_batch_start_valid
			end_point = np.minimum(start_point + self.batch_size, self.x_valid.shape[0])
			x_batch = self.x_valid[start_point : end_point]
			y_batch = self.y_valid[start_point : end_point]
			self.current_batch_start_valid = end_point
		elif split == 'Test' :
			start_point = self.current_batch_start_test
			end_point = np.minimum(start_point + self.batch_size, self.x_test.shape[0])
			x_batch = self.x_test[start_point : end_point]
			y_batch = self.y_test[start_point : end_point]
			self.current_batch_start_test = end_point

		# This place should never be reached by any code ...	
		else :
			print('[ERROR] Wrong split is querried : ', str(split))
			print('[ERROR] Terminating the code ...')
			sys.exit()

		return x_batch, y_batch


	# Define a method to get the entire data splits
	def get_data_split(self, split = 'Train') :

		"""
		inputs :

		split : 'Train'
			The split in which we want to check if the next batch exists
		"""

		"""
		outputs :

		x_split (implicit) :
			The split of the data. SHAPE : [<batch_size>, <feat_dim>]
		y_split (implicit) :
			The split of the labels. SHAPE : [<batch_size>, ]
		"""

		if split == 'Train' :
			return self.x_train, self.y_train
		elif split == 'Valid' :
			return self.x_valid, self.y_valid
		elif split == 'Test' :
			return self.x_test, self.y_test
		# This place should never be reached by any code ...	
		else :
			print('[ERROR] Wrong split is querried : ', str(split))
			print('[ERROR] Terminating the code ...')
			sys.exit(status)


	# Define a function to reset the batch generation
	def reset_data_split(self, split = 'Train') :

		"""
		inputs :

		split : 'Train'
			The split in which we want to check
		"""

		"""
		outputs :
		"""

		if split == 'Train' :
			self.current_batch_start_train = 0
		elif split == 'Valid' :
			self.current_batch_start_valid = 0
		elif split == 'Test' :
			self.current_batch_start_test = 0
		# This place should never be reached by any code ...	
		else :
			print('[ERROR] Wrong split is querried : ', str(split))
			print('[ERROR] Terminating the code ...')
			sys.exit(status)


# Pseudo-main
if __name__ == '__main__' :

	# Create the .npy files
	overwrite_npy(dataset_path = './')

	# Create a dataset instance
	binarized_mnist = BinarizedMNIST(	batch_size = 1, 
										dataset_path = './', 
										is_load_from_npy = True, 
										is_load_from_amat_files = False)

	# Visualize a batch each
	train_batch_x, train_batch_y = binarized_mnist.get_next_batch(split = 'Train')
	valid_batch_x, valid_batch_y = binarized_mnist.get_next_batch(split = 'Valid')
	test_batch_x, test_batch_y = binarized_mnist.get_next_batch(split = 'Test')

	# Print shapes
	print('[DEBUG] Shape of train batch : ', train_batch_x.shape)
	print('[DEBUG] Shape of valid batch : ', valid_batch_x.shape)
	print('[DEBUG] Shape of test batch : ', test_batch_x.shape)
	print('[DEBUG] Image Entries Range : [', np.min(train_batch_x), ', ', np.max(train_batch_y),']')
	print('[DEBUG] Image Batch Shape : ', train_batch_x.shape)
	sys.exit()

	# Visualization specific only
	import Display_Custom_Plots as display_utils
	import Print_Updating_Info as print_utils
	# Create a screen
	print_screen = print_utils.screenToPrintUpdatingInfo()
	display_screen = display_utils.screenToDisplayCustomPlots()

	display_screen.DisplayImagesFromSourcesInCustomPlot(	print_screen = print_screen, 
															image_sources = train_batch_x.reshape([-1, 28, 28, 1]))

	display_screen.DisplayImagesFromSourcesInCustomPlot(	print_screen = print_screen, 
															image_sources = valid_batch_x.reshape([-1, 28, 28, 1]))

	display_screen.DisplayImagesFromSourcesInCustomPlot(	print_screen = print_screen, 
															image_sources = test_batch_x.reshape([-1, 28, 28, 1]))
