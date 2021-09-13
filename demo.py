from stego import stego_block

import numpy as np
from PIL import Image

image = stego_block.image_read('/home/zgebac26/python/stego_project/assets/test_img.jpg')
print(image.shape)

k_channel = stego_block.extract_k_channel(image)
print(k_channel.shape)

blocks = stego_block.image_to_blocks(k_channel, 2)
print(blocks.shape)

fourier_magnitude, fourier_phase = stego_block.image_to_fourier(blocks)
print(fourier_magnitude.shape)
print(fourier_phase.shape)

# Saving image
imgObject = Image.fromarray(fourier_magnitude[:, :, 1].astype('uint8'))
imgName = '/home/zgebac26/python/stego_project/assets/fourier_magnitude.jpg'
imgObject.save(imgName)

imgObject = Image.fromarray(fourier_phase[:, :, 1].astype('uint8'))
imgName = '/home/zgebac26/python/stego_project/assets/fourier_phase.jpg'
imgObject.save(imgName)