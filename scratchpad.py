from PIL import Image, ImageCms
import skimage.metrics as msr
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import math
from stego import stego_block

radius = 1/4 * 512

print(np.around((radius*math.cos(math.pi/4))))
print(np.around((radius*math.sin(math.pi/4))))

# Saving image
# imgObject = Image.fromarray(img.astype('uint8'), 'CMYK')
# imgObject.save(imgName)
# print(f"Save merged image {img.shape}")