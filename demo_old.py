from wmark import WaterMark

import numpy as np
from PIL import Image

# Initializing the class with a seed
w_embed = WaterMark(5)
w_decode = WaterMark(15)

# Reading image
img = WaterMark.imread('/home/zgebac26/python/stego_project/img_orig.tif')
print(f"Read {img.shape} image as {img.dtype}...")

# Embedding data
marked = w_embed.embedMark(img, factor = 1000)
print(f"Mark {marked.shape} marked channel as {marked.dtype}...")

# Saving image
imgObject = Image.fromarray(marked.astype('uint8'), 'CMYK')
imgName = '/home/zgebac26/python/stego_project/img_marked_ante.jpg'
imgObject.save(imgName)
print(f"Save marked image {marked.shape}")

# Saving image
# imgObject = Image.fromarray(marked[:, :, 0].astype('uint8'), 'L')
# imgName = '/Users/zgebac/Desktop/grf-projekt/python/stego_project/img_marked_channel_ante.jpg'
# imgObject.save(imgName)
# print("Done!")

# Decoding data
wrong = w_decode.decodeMark(marked, metric = 'CORR')
correct = w_embed.decodeMark(marked, metric = 'CORR')

print(f"Decode value for correct seed: {round(correct, 3)}")
print(f"Decode value for wrong seed: {round(wrong,3)}")