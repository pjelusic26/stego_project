import time
start = time.time()

print(f"Started script...")

from stego import stego_block

import numpy as np
import pandas as pd
from PIL import Image
import skimage.metrics as msr
import glob
import os
from pathlib import Path

# Define source folder for image and color profile
img_path = 'test_images/img-0205.tif'
profile_path = 'profiles/ISOcoated_v2_eci.icc'

# Instantiate stego object
stego = stego_block(5, 5, 5)

# Create implementation strength range
implementation_range = np.arange(0, 30.1, 1.0)
# implementation_range = np.arange(5, 5.1, 1)

# Create empty array
results = np.zeros((len(implementation_range), 11))

# Read, resize image and extract channel
img = stego_block.image_read(img_path)
resized = stego_block.image_resize(img, 512)
channel = stego_block.extract_channel(resized)

# Save original channel and image
# stego.image_save(resized.astype('uint8'), 'images/spatial_orig.jpg', 'CMYK')
# stego.image_save(channel.astype('uint8'), 'images/spatial_orig_k.jpg', 'L')

counter = 0

for i in implementation_range:
    # Zero-marking
    marked_zero = np.copy(resized)
    marked_zero[:, :, -1] = stego.embed_data('A', marked_zero[:, :, -1], 200, (3, 3), 'M', 0)[0]
    # Real embedding
    # marked_l = stego.embed_data('A', channel, 200, (3, 3), 'L', i)
    marked_lm = stego.embed_data('A', channel, 200, (3, 3), 'LM', i)
    marked_m = stego.embed_data('A', channel, 200, (3, 3), 'M', i)
    marked_mh = stego.embed_data('A', channel, 200, (3, 3), 'MH', i)
    # marked_h = stego.embed_data('A', channel, 200, (3, 3), 'H', i)
    # Merge image channels
    # merged_l = stego.image_merge_channels(resized, marked_l[0])
    merged_lm = stego.image_merge_channels(resized, marked_lm[0])
    merged_m = stego.image_merge_channels(resized, marked_m[0])
    merged_mh = stego.image_merge_channels(resized, marked_mh[0])
    # merged_h = stego.image_merge_channels(resized, marked_h[0])
    # Apply GCR masking
    # masked_l = stego.gcr_masking(resized, merged_l, profile_path)
    masked_lm = stego.gcr_masking(resized, merged_lm, profile_path)
    masked_m = stego.gcr_masking(resized, merged_m, profile_path)
    masked_mh = stego.gcr_masking(resized, merged_mh, profile_path)
    # masked_h = stego.gcr_masking(resized, merged_h, profile_path)
    # Calculate PSNR values
    # psnr_l = stego.lab_quality(marked_zero, merged_l, profile_path, metric = 'SSIM')
    # psnr_masked_l = stego.lab_quality(marked_zero, masked_l, profile_path, metric = 'SSIM')
    # psnr_lm = stego.lab_quality(marked_zero, merged_lm, profile_path, metric = 'SSIM')
    # psnr_masked_lm = stego.lab_quality(marked_zero, masked_lm, profile_path, metric = 'SSIM')
    # psnr_m = stego.lab_quality(marked_zero, merged_m, profile_path, metric = 'SSIM')
    # psnr_masked_m = stego.lab_quality(marked_zero, masked_m, profile_path, metric = 'SSIM')
    # psnr_mh = stego.lab_quality(marked_zero, merged_mh, profile_path, metric = 'SSIM')
    # psnr_masked_mh = stego.lab_quality(marked_zero, masked_mh, profile_path, metric = 'SSIM')
    # psnr_h = stego.lab_quality(marked_zero, merged_h, profile_path, metric = 'SSIM')
    # psnr_masked_h = stego.lab_quality(marked_zero, masked_h, profile_path, metric = 'SSIM')
    # Extract marked channel
    # channel_l = stego.extract_channel(masked_l)
    channel_lm = stego.extract_channel(masked_lm)
    channel_m = stego.extract_channel(masked_m)
    channel_mh = stego.extract_channel(masked_mh)
    # channel_h = stego.extract_channel(masked_h)
    # Decode pattern
    # decode_l = stego.grubbs_test('A', channel_l, 200, 'L', alpha = 0.05)

    decode_lm = stego.grubbs_test('A', channel_lm, 200, 'LM', alpha = 0.05)
    decode_lm_false = stego.grubbs_test('B', channel_lm, 200, 'LM', alpha = 0.05)

    decode_m = stego.grubbs_test('A', channel_m, 200, 'M', alpha = 0.05)
    decode_m_false = stego.grubbs_test('B', channel_m, 200, 'M', alpha = 0.05)

    decode_mh = stego.grubbs_test('A', channel_mh, 200, 'MH', alpha = 0.05)
    decode_mh_false = stego.grubbs_test('B', channel_mh, 200, 'MH', alpha = 0.05)

    # decode_h = stego.grubbs_test('A', channel_h, 200, 'H', alpha = 0.05)

    # print(f"Freq L  Grubbs for alpha {i} says: {decode_l}")
    print(f"True  Freq LM Grubbs for alpha {i} says: {decode_lm}")
    print(f"False Freq LM Grubbs for alpha {i} says: {decode_lm_false}")

    print(f"True  Freq M  Grubbs for alpha {i} says: {decode_m}")
    print(f"False Freq M  Grubbs for alpha {i} says: {decode_m_false}")

    print(f"True  Freq MH Grubbs for alpha {i} says: {decode_mh}")
    print(f"False Freq MH Grubbs for alpha {i} says: {decode_mh_false}")

    # print(f"Freq H  Grubbs for alpha {i} says: {decode_h}")

    # results[counter, :] = [
    #     i, 
    #     psnr_l, psnr_masked_l, 
    #     psnr_lm, psnr_masked_lm, 
    #     psnr_m, psnr_masked_m, 
    #     psnr_mh, psnr_masked_mh,
    #     psnr_h, psnr_masked_h
    # ]

    counter += 1
    print(i)


# df = pd.DataFrame(results)
# df.to_csv('five_frequencies_0_ssim.csv')

# Saving image
stego.image_save(merged_l, 'images/spatial_marked_005.jpg', 'CMYK')
stego.image_save(masked_l, 'images/spatial_masked_005.jpg', 'CMYK')

end = time.time()
print(
    f"Done with script in {(end - start) / 60} minutes."
)
