from PIL import Image, ImageCms
import skimage.metrics as msr
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from stego import stego_block

path = Path(__file__).resolve()
image_path = str(path.parents[1])+'/stego_project/test_set/coated/orig/img_01.tif'
imgName = '/home/zgebac26/python/stego_project/test_set/coated/orig/img_01_590.tif'

# Create wmark object with seed 5
stego_object = stego_block(5, 5, 5)

# Read image
img = stego_block.image_read(image_path)

# Resize image
img = stego_block.image_resize(img, 590)

# Define center of image
# quarter = int(img.shape[0] / 4)
# print(quarter)

# Define half of image
# half = int(img.shape[0] / 2)
# print(half)

# img_half = img[half:, :, :]
# print(f"Img Top: {img_half.shape}")

# Saving image
imgObject = Image.fromarray(img.astype('uint8'), 'CMYK')
imgObject.save(imgName)
print(f"Save merged image {img.shape}")