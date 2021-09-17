from stego import stego_block

import numpy as np
from PIL import Image

# Initializing the class with a seed
stego8 = stego_block(8)

# Reading image
print("Reading image...")
image = stego_block.image_read('/home/zgebac26/python/stego_project/test_img.tif')

# Resizing image
print("Resizing image...")
image = stego_block.image_resize(image, 1024)

print("Extracting channel...")
# Extracting channel
channel = stego_block.extract_channel(image)

print("Splitting image to blocks...")
# Dividing image to blocks
blocks = stego_block.image_to_blocks(channel, 2)

print("Embedding data to blocks...")
# Embedding data into each block separately
blocks_marked = np.copy(blocks)
counter = 0
for i in range(blocks.shape[-1]):
    blocks_marked[:, :, counter] = stego8.embed_data(blocks[:, :, counter], 
        vector_length = 200, frequency = 'MEDIUM', implementation_strength = 2000)
    counter += 1

print("Merging blocks...")
# Merging blocks
merged = stego_block.image_merge_blocks(blocks_marked, 2)

print("Merging channels...")
# Merging color channels
color = stego_block.image_merge_channels(image, merged)

print("Saving image...")
# Saving image
imgObject = Image.fromarray(color.astype('uint8'), 'CMYK')
imgName = '/home/zgebac26/python/stego_project/image_color.jpg'
imgObject.save(imgName)
print("Done!")