from stego import stego_block

import numpy as np
import pandas as pd
from PIL import Image

import glob
import os
from pathlib import Path

# Define source folder
src_folder = 'test_images/'

# Define source path
src_path = Path(src_folder).resolve()

# Define all images in folder
img_batch = src_path.glob('*tif')

# Define stego object
stego = stego_block(5, 5, 5)

# Create empty Pandas DF
df = pd.DataFrame(columns = ['Img', 'Factor', 'PSNR'])

# Just for nicer printing statements
nl = "\n"

for i in img_batch:
    img = stego_block.image_read(str(i))
    resized = stego_block.image_resize(img, 512)
    channel = stego_block.extract_channel(resized)
    factor = stego.implementation_strength('A', channel, (38, 42), 200, 'MEDIUM')
    print(i.name)
    df = df.append({"Img": i.name, "Factor": factor[0], "PSNR": factor[1]}, ignore_index = True)

print(df.head)
df.to_csv('impact_factor.csv')