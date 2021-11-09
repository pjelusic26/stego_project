from stego import image_processing, stego_block

import numpy as np
from PIL import Image

# New line
nl = "\n"

# Initializing image processing class
img_proc = image_processing(
    block_number = 2, 
    image_dimension = 1024
)

# Initializing stego class
stego_5 = stego_block(
    seed_pattern = 8, 
    seed_message = 16, 
    seed_permutation = 24, 
    block_number = 4,
    bit_amount = 3, 
    length = 200, 
    frequency = 'MEDIUM'
)
print(f"Initialized {stego_5} class.")

print(stego_5.generate_message())
print(stego_5.generate_permutation())

### IMG PRE PROC ###
### IMG PRE PROC ###
### IMG PRE PROC ###

img = image_processing.image_read('/home/zgebac26/python/stego_project/test_images/img-0004.tif')
print(f"Read {img.shape} image as {img.dtype}...")

img = img_proc.image_resize(img)
print(f"Resize image to {img.shape} as {img.dtype}...")

channel = image_processing.extract_channel(img)
print(f"Extract {channel.shape} channel as {channel.dtype}...")

blocks = img_proc.image_to_blocks(channel)
print(f"Divide channel to {blocks.shape} blocks...")

### STEGO ###
### STEGO ###
### STEGO ###

# Embedding data
marked = stego_5.embed_pattern_to_blocks(image_blocks = blocks, factor = 5)
print(f"Mark blocks {marked.shape} as {marked.dtype}...")

# Decoding data
# values = stego8.decode_data_from_blocks(marked, length = 200, frequency = 'MEDIUM')
# print(f"Decode values for blocks are:{nl}{values}")

# Performing Grubbs' outlier test
print(f"Performing Grubbs' test in {marked.shape} image...")
grubbs = stego_5.decode_data_pattern(marked, alpha = 0.05)
print(f"Grubbs says:{nl}{grubbs[1]}")

### IMG POST PROC ###
### IMG POST PROC ###
### IMG POST PROC ###

# Merging blocks
blocks = img_proc.image_merge_blocks(marked)
print(f"Merging blocks {blocks.shape} as {blocks.dtype}...")

# Merging channels
merged = image_processing.image_merge_channels(img, blocks)
print(f"Merge {merged.shape} image channels as {merged.dtype}...") 

# Saving image
image_processing.image_save(merged, '/home/zgebac26/python/stego_project/img_marked_pbj.jpg', 'CMYK')
# imgObject = Image.fromarray(merged.astype('uint8'), 'CMYK')
print(f"Save merged image {merged.shape}")