from PIL import Image, ImageCms
import skimage.metrics as msr
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from stego import stego_block

path = Path(__file__).resolve()
image_path = str(path.parents[1])+'/stego_project/test_set/uncoated/img_02.tif'
imgName = '/home/zgebac26/python/stego_project/test_set/uncoated/img_02_psnr_35_40.tif'

# Create wmark object with seed 5
stego_object = stego_block(5, 5, 5)

# Read and image
img = stego_block.image_read(image_path)

# Extract channel
channel = stego_block.extract_channel(img)

# Find appropriate implementation factor to get quality (PSNR)
factor = stego_object.implementation_strength('A', channel, (35, 40), 200, 'MEDIUM')
print(f"Factor: {factor[0]} PSNR: {factor[1]}")

# Embed mark in an image
channel_marked = stego_object.embed_data('A', channel, 200, 'MEDIUM', factor[0])

# Merge channels
img_marked = stego_block.image_merge_channels(img, channel_marked)

psnr_value = msr.peak_signal_noise_ratio(img[:, :, 0:3], img_marked[:, :, 0:3])
print(psnr_value)

# Saving image
imgObject = Image.fromarray(img_marked.astype('uint8'), 'CMYK')
imgObject.save(imgName)
print(f"Save merged image {img_marked.shape}")