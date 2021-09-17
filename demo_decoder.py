from stego import stego_block

import numpy as np

# Initializing the class with a seed
stego8 = stego_block(8)

print("Reading image...")
# Reading image
img = stego_block.image_read('/home/zgebac26/python/stego_project/image_color.jpg')

print("Extracting channel...")
# Extracting channel
channel = stego_block.extract_channel(img)

print("Splitting image to blocks...")
# Dividing image to blocks
blocks = stego_block.image_to_blocks(channel, 2)

print("Decoding data from blocks...")
# Decoding data from each block separately
blocks_marked = np.zeros((4, 1))
counter = 0
for i in range(blocks.shape[-1]):
    blocks_marked[counter] = stego8.decode_data(blocks[:, :, counter], length = 200, frequency = 'MEDIUM')
    counter += 1

print(blocks_marked) 