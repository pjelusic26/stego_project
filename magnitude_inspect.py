from PIL import Image, ImageCms
import skimage.metrics as msr
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from stego import stego_block

path = Path(__file__).resolve()
image_path = str(path.parents[1])+'/stego_project/test_set/uncoated/img_03.tif'

# Create wmark object with seed 5
stego_object = stego_block(5, 5, 5)

# Read and image
img = stego_block.image_read(image_path)

# Extract channel
channel = stego_block.extract_channel(img)
print(f"Max  spatial: {round(np.amax(channel), 3)}")
print(f"Min  spatial: {round(np.amin(channel), 3)}")
print(f"Mean spatial: {round(np.mean(channel), 3)}")
print(f"Med  spatial: {round(np.median(channel), 3)}")

# Get magnitude and phase
magnitude, phase = stego_object.image_to_fourier(channel)
print(f"Max  magnitude: {round(np.amax(magnitude), 3)}")
print(f"Min  magnitude: {round(np.amin(magnitude), 3)}")
print(f"Mean magnitude: {round(np.mean(magnitude), 3)}")
print(f"Med  magnitude: {round(np.median(magnitude), 3)}")

# Embed with IF 2000
marked = stego_object.embed_data('A', channel, 200, 'MEDIUM', 2000)

# Calculate PSNR
psnr = msr.peak_signal_noise_ratio(channel, marked)
print(f"PSNR: {round(psnr, 3)}")