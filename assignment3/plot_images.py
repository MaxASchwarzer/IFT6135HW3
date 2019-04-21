# Dependencies
import os
import numpy as np

from Display_Custom_Plots import screenToDisplayCustomPlots
from Print_Updating_Info import screenToPrintUpdatingInfo


# Create a screen
print_screen = screenToPrintUpdatingInfo()
plot_screen = screenToDisplayCustomPlots()

# Dataset directory
image_dir = './'

# Load the path
# image_path = image_dir + 'x_gen_SVHN.npy'
# image_path = image_dir + 'x_gen_BMNIST.npy'
images = np.load(image_path)
print(images.shape)
images = np.transpose(images, (0, 2, 3, 1))
print(images.shape)

indices = np.random.randint(low = 0, high = images.shape[0], size = 100)

plot_screen.DisplayImagesFromSourcesInCustomPlot(print_screen = print_screen, image_sources = images[indices])