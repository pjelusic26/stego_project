from stego import stego_block

import numpy as np

# Reading image
img = stego_block.image_read('/home/zgebac26/python/stego_project/image_color.jpg')
print("Reading image...")

# Extracting channel
channel = stego_block.extract_channel(img)
print("Extracting channel...")

# Dividing image to blocks
blocks = stego_block.image_to_blocks(channel, 4)
print("Splitting image to blocks...")

# Decoding data from each block separately
blocks_marked = np.zeros((16, 1))
counter = 0

for i in range(blocks.shape[-1]):
    blocks_marked[counter] = stego_block.decode_data(blocks[:, :, counter], length = 200, frequency = 'MEDIUM')
    counter += 1
print("Decoding data from blocks...")

print(blocks_marked)