from stego import stego_block
from wmark import WaterMark

import numpy as np
from PIL import Image

# Initializing the class with a seed
w_5 = stego_block(5, 15)

# New line
nl = "\n"

# Reading image
img = w_5.image_read('/home/zgebac26/python/stego_project/img_orig.tif')
print(f"Read {img.shape} image as {img.dtype}...")

# Resize image
img = w_5.image_resize(img, 512)
print(f"Resize image to {img.shape} as {img.dtype}...")

# Extracting channel
channel = w_5.extract_channel(img)
print(f"Extract {channel.shape} channel as {channel.dtype}...")

# Finding the optimal implementation strength
factor = w_5.implementation_strength(key_choice = 'A', img_block = channel)
print(f"Optimal factor is: {factor[0]}")
print(f"PSNR value is: {factor[1]}")

# Embedding data
marked = w_5.embed_data(key_choice = 'A', img_channel = channel, length = 200, frequency = 'MEDIUM', factor = factor[0])
print(f"Mark blocks {marked.shape} as {marked.dtype}...")

# Grubbs outlier test
grubbs = w_5.grubbs_test(key_choice = 'A', img_block = marked, length = 200, frequency = 'MEDIUM', alpha = 0.05)
print(f"Grubbs says {grubbs[0]}")
print(f"Grubbs had these values to make the decision:{nl}{grubbs[1]}")

