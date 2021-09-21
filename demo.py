from stego import stego_block
from wmark import WaterMark

import numpy as np
from PIL import Image

# Initializing the class with a seed
stego8 = stego_block(5)
stegoAnte = WaterMark(5)

print("Reading image...")
# Reading image
img = stego_block.image_read('/Users/zgebac/Desktop/grf-projekt/python/stego_project/img_greyscale.png')
print(img.dtype)

print("Extracting channel...")
# Extracting channel
channel = stego_block.extract_channel(img)
print(channel.dtype)

print("Embedding data...")
# Embedding data
marked = stego8.embed_data(channel, vector_length = 200, frequency = 'MEDIUM', implementation_strength = 1000)
print(marked.dtype)

print("Saving image...")
# Saving image
imgObject = Image.fromarray(marked.astype('uint8'), 'L')
imgName = '/Users/zgebac/Desktop/grf-projekt/python/stego_project/img_marked_channel_pbj.jpg'
imgObject.save(imgName)
print("Done!")

print("Merging channels...")
# Merging channels
merged = stego_block.image_merge_channels(img, marked)
print(merged.dtype)

print("Decoding data...")
# Decoding data
correct = stego8.decode_data(marked, length = 200, frequency = 'MEDIUM')
wrong = stegoAnte.decodeMark(merged, metric = 'CORR')

print(correct)
print(wrong)

print("Saving image...")
# Saving image
imgObject = Image.fromarray(merged.astype('uint8'), 'L')
imgName = '/Users/zgebac/Desktop/grf-projekt/python/stego_project/img_marked_pbj.jpg'
imgObject.save(imgName)
print("Done!")