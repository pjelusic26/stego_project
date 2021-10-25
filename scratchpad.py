from PIL import Image, ImageCms
import skimage.metrics as msr
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import math
from stego import stego_block

radius = 1/4 * 512
img_channel = 512

x1 = int((img_channel/2) + np.around(radius*math.cos(math.pi/1)))
y1 = int((img_channel/2) + np.around(radius*math.sin(math.pi/1)))

x2 = int((img_channel/2) + np.around(radius*math.cos(math.pi/1 + math.pi/8)))
y2 = int((img_channel/2) + np.around(radius*math.sin(math.pi/1 + math.pi/8)))

print('Done!')
# Saving image
# imgObject = Image.fromarray(img.astype('uint8'), 'CMYK')
# imgObject.save(imgName)
# print(f"Save merged image {img.shape}")