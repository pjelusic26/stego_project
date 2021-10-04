from stego import stego_block

import numpy as np
from PIL import Image

# Initializing the class with a seed
stego8 = stego_block(seed_pattern = 2, seed_message = 4, seed_permutation = 8)

# New line
nl = "\n"

# Generating permutation
permutation = stego8.generate_permutation(4)
print(f"Decoder permutation:{nl}{permutation}")

# Reading image from Ofiice Linux PC
# img = stego_block.image_read('/home/zgebac26/python/stego_project/img_marked_pbj.jpg')
# Reading image from MacBook
img = stego_block.image_read('/Users/zgebac/Desktop/grf-projekt/python/stego_project/img_channel.jpg')

# Extracting channel
channel = stego_block.extract_channel(img)

# Dividing image to blocks
blocks = stego_block.image_to_blocks(channel, 2)

# Decoding data from each block separately
decoded_values, decoded_message = stego8.decode_data_pattern(
    permutation,
    blocks,
    length = 200, 
    frequency = 'MEDIUM', 
    alpha = 0.001)
print(f"Decoded message:{nl}{decoded_message}")