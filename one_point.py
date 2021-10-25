from stego import stego_block

import numpy as np
import pandas as pd
from PIL import Image
import skimage.metrics as msr
import glob
import os
from pathlib import Path

# Define source folder
img_path = 'test_set/img-0001.tif'

# Define stego object
stego = stego_block(5, 5, 5)

# Create implementation strength range
implementation_range = range(0, 80001, 40)
# implementation_range = range(0, 26, 25)

# Create empty array
results = np.zeros((len(implementation_range), 10))

# Read image and extract channel
img = stego_block.image_read(img_path)
resized = stego_block.image_resize(img, 512)
channel = stego_block.extract_channel(resized)

counter = 0

for i in implementation_range:
    marked_low = stego.embed_data('A', channel, 1, (1, 1), 'LOW', i)
    psnr_low = msr.peak_signal_noise_ratio(channel, marked_low[0], data_range = 255)

    marked_medium = stego.embed_data('A', channel, 1, (1, 1), 'MEDIUM', i)
    psnr_medium = msr.peak_signal_noise_ratio(channel, marked_medium[0], data_range = 255)

    marked_high = stego.embed_data('A', channel, 1, (1, 1), 'HIGH', i)
    psnr_high = msr.peak_signal_noise_ratio(channel, marked_high[0], data_range = 255)

    results[counter, :] = [i, marked_low[1], marked_low[2], psnr_low, marked_medium[1], marked_medium[2], psnr_medium, marked_high[1], marked_high[2], psnr_high]

    counter += 1
    print(i)

df = pd.DataFrame(results)
df.to_csv('one_point.csv')

print(f"Done!")
