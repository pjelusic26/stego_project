### Stego project from scratch ###

import numpy as np
from PIL import Image
from scipy import fftpack

### List of methods to do ###

class stego_block:

    ### Encoder ###
    ### Encoder ###
    ### Encoder ###

    # 1. Read cmyk image
    def image_read(filepath):
        img_orig = np.array(Image.open(filepath))
        return img_orig

    # 2. Choose K channel
    def extract_k_channel(img_orig):
        if len(img_orig.shape) == 2:
            return img_orig
        elif len(img_orig.shape) == 3:
            if img_orig.shape[-1] == 3:
                return img_orig[:, :, 0]
            elif img_orig.shape[-1] == 4:
                return img_orig[:, :, -1]
        else:
            raise AttributeError("Please provide grayscale, RGB or CMYK image.")

    # 3. Split image into blocks
    def image_to_blocks(img_k, block_number):
        
        block_height = int(img_k.shape[0] / block_number)
        block_width = int(img_k.shape[1] / block_number)
        block_depth = int(block_number ** 2)

        blocks_output = np.zeros((int(block_height), int(block_width), int(block_depth)))

        divide_a = np.split(img_k, block_number, axis = 0)

        counter = 0

        for i in range(block_number):
            for j in range(block_number):

                divide_b = np.split(divide_a[i], block_number, axis = 1)
                blocks_output[:, :, counter] = divide_b[j]
                counter += 1

        return blocks_output

    # 4. Test activity levels for each block
    def activity_test(img_k_blocks, activity_threshold):
        return activity_map

    # 5. Apply Fourier transform
    def image_to_fourier(img_k_blocks):
        
        img_magnitude = np.copy(img_k_blocks)
        img_phase = np.copy(img_k_blocks)
        counter = 0

        for i in range(img_k_blocks.shape[-1]):
            img_magnitude[:, :, counter] = fftpack.fft2(img_k_blocks[:, :, counter])
            img_magnitude[:, :, counter] = fftpack.fftshift(np.absolute(img_magnitude[:, :, counter]))

            img_phase[:, :, counter] = np.angle(img_magnitude[:, :, counter])

            counter += 1

        return img_magnitude, img_phase

    # 6. Embed watermarks
    def embed_data(img_fourier, secret_key, implementation_strength):
        return img_fourier_marked

    # 7. Apply inverse Fourier transform
    def image_to_spatial(img_fourier_marked):
        return img_marked

    # 8. Apply GCR Masking
    def gcr_masking(img_marked, gcr_method, color_profile):
        return img_masked

    # 9. Merge blocks
    def image_merge_blocks(img_masked, block_number):
        return img_merged

    # 10. Merge marked K channel with original C M Y
    def image_merge_channels(img_orig, img_merged):
        return img_output

    ### Decoder ###
    ### Decoder ###
    ### Decoder ###

    # 1. Read cmyk image
    # 2. Split image into blocks
    # 3. Choose K channels
    # 4. Activity test
    # 5. Apply Fourier transform
    # 6. Search for watermarks
    def decode_data(img_output):
        return decode_values

    # 7. Grubbs' test
    def grubbs_test(decode_values, alpha):
        return grubbs_values

    # 8. Extract message (matched pattern)
    def pattern_matching(grubbs_values):
        return pattern_output