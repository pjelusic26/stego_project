import time
import skimage.metrics as msr
import numpy as np
from pathlib import Path
import pandas as pd

from stego import stego_block

time_start = time.time()

# Source Directory
srcFolder = 'test_set/'
# Source Path
srcPth = Path(srcFolder).resolve()
# Define all .tif images in folder
imgs = srcPth.glob('*.tif')

# Create wmark object with seed 5
stego_object = stego_block(5, 5, 5)

# Magic numbers
dataset_size = 1001
feature_set_size = 11
impact_factor = 1000

print(f"Found {dataset_size} images in folder.")
print(f"Using implementation strength {impact_factor}...")
print(f"Saving {feature_set_size} features...")

# Features to extract
# array_features = np.zeros((dataset_size, feature_set_size))
array_features = np.empty((dataset_size, feature_set_size))

# Start counting
counter = 0

for i in imgs:

    # Read and image
    img = stego_block.image_read(i)

    # Extract channel
    channel = stego_block.extract_channel(img)

    # Mark image
    marked = stego_object.embed_data('A', channel, 200, 'MEDIUM', impact_factor)

    # Get magnitude and phase
    magnitude, phase = stego_object.image_to_fourier(channel)

    # Calculate PSNR
    psnr = msr.peak_signal_noise_ratio(channel, marked)

    # Image name
    # filename = str(i.name)
    # filename = Path(filename).stem
    # print(filename)

    # Standard deviation
    spatial_std = np.std(channel)
    # Mean pixel value
    spatial_mean = np.mean(channel)
    # Median pixel value
    spatial_median = np.median(channel)
    # Max pixel value
    spatial_max = np.amax(channel)
    # Min pixel value
    spatial_min = np.amin(channel)

    # Magnitude deviation
    magnitude_std = np.std(magnitude)
    # Mean magnitude
    magnitude_mean = np.mean(magnitude)
    # Median magnitude
    magnitude_median = np.median(magnitude)
    # Max magnitude
    magnitude_max = np.amax(magnitude)
    # Min magnitude
    magnitude_min = np.amin(magnitude)

    # array_features[counter, 0] = str(filename)
    array_features[counter, 0] = spatial_std
    array_features[counter, 1] = spatial_mean
    array_features[counter, 2] = spatial_median
    array_features[counter, 3] = spatial_max
    array_features[counter, 4] = spatial_min
    array_features[counter, 5] = magnitude_std
    array_features[counter, 6] = magnitude_mean
    array_features[counter, 7] = magnitude_median
    array_features[counter, 8] = magnitude_max
    array_features[counter, 9] = magnitude_min
    array_features[counter, 10] = psnr

    counter += 1
    print(counter)

data_frame = pd.DataFrame(array_features)
data_frame.columns = ['Spatial STD', 'Spatial Mean', 'Spatial Median', 'Spatial Max', 'Spatial Min', 'Magnitude STD', 'Magnitude Mean', 'Magnitude Median', 'Magnitude Max', 'Magnitude Min', 'PSNR']
data_frame.to_csv('image_features.csv')

time_end = time.time()
print(time_end - time_start)