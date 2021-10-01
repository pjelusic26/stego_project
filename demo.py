from stego import stego_block
from wmark import WaterMark

import numpy as np
from PIL import Image

# Initializing the class with a seed
# stego8 = stego_block(8)
# stego12 = stego_block(12)
stego8 = stego_block(8, 12)

# New line
nl = "\n"

# Reading image
img = stego_block.image_read('/home/zgebac26/python/stego_project/img_orig.tif')
print(f"Read {img.shape} image as {img.dtype}...")

# Resize image
img = stego_block.image_resize(img, 1024)
print(f"Resize image to {img.shape} as {img.dtype}...")

# Extracting channel
channel = stego_block.extract_channel(img)
print(f"Extract {channel.shape} channel as {channel.dtype}...") 

# Dividing image to blocks
blocks = stego_block.image_to_blocks(channel, 2)
print(f"Divide channel to {blocks.shape} blocks...")

# Embedding data
marked = stego8.embed_data_to_blocks(image_blocks = blocks, length = 200, frequency = 'MEDIUM', factor = 1000)
print(f"Mark blocks {marked.shape} as {marked.dtype}...")

# Decoding data
# values = stego8.decode_data_from_blocks(marked, length = 200, frequency = 'MEDIUM')
# print(f"Decode values for blocks are:{nl}{values}")

# Performing Grubbs' outlier test
grubbs = stego8.pattern_matching(marked, length = 200, frequency = 'MEDIUM', alpha = 0.05)
print(f"Grubbs says:{nl}{grubbs}")

# Merging blocks
blocks = stego_block.image_merge_blocks(marked, 2)
print(f"Merging blocks {blocks.shape} as {blocks.dtype}...")

# Merging channels
merged = stego_block.image_merge_channels(img, blocks)
print(f"Merge {merged.shape} image channels as {merged.dtype}...")

# Saving image
imgObject = Image.fromarray(merged.astype('uint8'), 'CMYK')
imgName = '/home/zgebac26/python/stego_project/img_marked_pbj.jpg'
imgObject.save(imgName)
print(f"Save merged image {merged.shape}")