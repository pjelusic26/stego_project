from stego import stego_block

import numpy as np
from PIL import Image

# Reading image
image = stego_block.image_read('/home/zgebac26/python/stego_project/test_img.jpg')
print("Reading image...")

# Extracting channel
channel = stego_block.extract_channel(image)
print("Extracting channel...")

# Dividing image to blocks
blocks = stego_block.image_to_blocks(channel, 4)
print("Splitting image to blocks...")

# Embedding data into each block separately
blocks_marked = np.copy(blocks)
counter = 0

for i in range(blocks.shape[-1]):
    blocks_marked[:, :, counter] = stego_block.embed_data(blocks[:, :, counter], 
        vector_length = 200, frequency = 'MEDIUM', implementation_strength = 5000)
    counter += 1
print("Embedding data to blocks...")

# Merging blocks
merged = stego_block.image_merge_blocks(blocks_marked, 4)
print("Merging blocks...")

# Merging color channels
color = stego_block.image_merge_channels(image, merged)
print("Merging channels...")

# Saving image
imgObject = Image.fromarray(color.astype('uint8'), 'CMYK')
imgName = '/home/zgebac26/python/stego_project/image_color.jpg'
imgObject.save(imgName)
print("Done!")