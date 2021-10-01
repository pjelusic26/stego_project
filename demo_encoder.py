from stego import stego_block

import numpy as np
from PIL import Image

# Initializing the class with a seed
stego8 = stego_block(seed_pattern = 2, seed_message = 4, seed_permutation = 8)

# New line
nl = "\n"

# Generating message
message = stego8.generate_message(6)
print(f"Encoder message:{nl}{message}")

# Generating permutation
permutation = stego8.generate_permutation(16)
print(f"Encoder permutation:{nl}{permutation}")

# Reading image
img = stego_block.image_read('/home/zgebac26/python/stego_project/test_img.jpg')

# Resize image
img = stego_block.image_resize(img, 2048)

# Extracting channel
channel = stego_block.extract_channel(img)

# Dividing image to blocks
blocks = stego_block.image_to_blocks(channel, 4)

# Embedding data to blocks
marked = stego8.embed_pattern_to_blocks(message, permutation, blocks, length = 200, frequency = 'MEDIUM', factor = 20000)

# Merging blocks
blocks = stego_block.image_merge_blocks(marked, 4)

# Saving image
imgObject = Image.fromarray(blocks.astype('uint8'), 'L')
imgName = '/home/zgebac26/python/stego_project/img_channel_marked_pbj.jpg'
imgObject.save(imgName)
print("Done!")

# Merging channels
merged = stego_block.image_merge_channels(img, blocks)

# Saving image
imgObject = Image.fromarray(merged.astype('uint8'), 'CMYK')
imgName = '/home/zgebac26/python/stego_project/img_marked_pbj.jpg'
imgObject.save(imgName)
print("Done!")