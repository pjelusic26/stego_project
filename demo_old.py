from wmark import WaterMark

import numpy as np
from PIL import Image

# Initializing the class with a seed
w_embed = WaterMark(5)
w_decode = WaterMark(15)

print("Reading image...")
# Reading image
img = WaterMark.imread('/Users/zgebac/Desktop/grf-projekt/python/stego_project/img_greyscale.png')
print(img.dtype)

print("Embedding data...")
# Embedding data
marked = w_embed.embedMark(img, factor = 1000)
print(marked.dtype)

print("Saving image...")
# Saving image
imgObject = Image.fromarray(marked.astype('uint8'), 'L')
imgName = '/Users/zgebac/Desktop/grf-projekt/python/stego_project/img_marked_ante.jpg'
imgObject.save(imgName)
print("Done!")

print("Saving image...")
# Saving image
# imgObject = Image.fromarray(marked[:, :, 0].astype('uint8'), 'L')
# imgName = '/Users/zgebac/Desktop/grf-projekt/python/stego_project/img_marked_channel_ante.jpg'
# imgObject.save(imgName)
# print("Done!")

print("Decoding data...")
# Decoding data
wrong = w_decode.decodeMark(marked, metric = 'CORR')
correct = w_embed.decodeMark(marked, metric = 'CORR')

print(correct)
print(wrong)