### Stego project from scratch ###

import numpy as np
from PIL import Image
from scipy import fftpack
import skimage
import skimage.color as color
import skimage.transform as trs
import math
import cv2

### Well, not exactly from scratch ###
# from wmark import WaterMark

class stego_block:

    def __init__(self, seed):
        self.seed = seed
        pass

    ### Encoder ###
    ### Encoder ###
    ### Encoder ###

    # 1. Read cmyk image
    @staticmethod
    def image_read(filepath):

        # Read image as Numpy array
        img_orig = np.array(Image.open(filepath))
        return img_orig

    # 2. Choose K channel
    @staticmethod
    def extract_channel(img_orig):

        # Checking if image is grayscale or color
        if len(img_orig.shape) == 2:
            return img_orig
        elif len(img_orig.shape) == 3:
            # Checking if color image is CMYK or RGB
            if img_orig.shape[-1] == 3:
                img_orig = color.rgb2ycbcr(img)
                return img_orig[:, :, 0]
            elif img_orig.shape[-1] == 4:
                return img_orig[:, :, -1]
        else:
            raise AttributeError("Please provide grayscale, RGB or CMYK image.")

    @staticmethod
    # 3. Split image into blocks
    def image_to_blocks(img_channel, block_number):
        
        # Defining block dimensions
        block_height = int(img_channel.shape[0] / block_number)
        block_width = int(img_channel.shape[1] / block_number)
        block_depth = int(block_number ** 2)

        # Creating empty array to be filled with image blocks
        blocks_output = np.zeros((int(block_height), int(block_width), int(block_depth)))

        # Split the image over the first axis
        divide_a = np.split(img_channel, block_number, axis = 0)

        counter = 0

        # Split image blocks over the other axis
        for i in range(block_number):
            for j in range(block_number):

                divide_b = np.split(divide_a[i], block_number, axis = 1)
                blocks_output[:, :, counter] = divide_b[j]
                counter += 1

        return blocks_output

    # TODO
    # 4. Test activity levels for each block
    # def activity_test(img_channel_blocks, activity_threshold):
    #     return activity_map

    # 5. Embed data
    def embed_data(self, img, vector_length, frequency, implementation_strength):

        # Extract channel from image
        img_channel = stego_block.extract_channel(img)

        # Get radius from provided frequency range
        radius = stego_block.vector_radius(img, frequency)

        # Generate mark using the secret key (seed)
        data_mark = self.key_generator(vector_length)

        # Transform the input image into the frequency domain
        magnitude, phase = self.image_to_fourier(img_channel)

        # Define the data mask (where the data will be embedded within the frequency domain)
        mark_mask = np.zeros(img_channel.shape)
        mask_shape = (3, 3)
        mask_up = np.zeros(mask_shape)
        mask_down = np.zeros(mask_shape)

        # Defining key points on the circular-shaped vector
        for ind in range(vector_length):
            x1 = int((img_channel.shape[0]/2) + np.around(radius*math.cos(ind*math.pi/vector_length)))
            y1 = int((img_channel.shape[0]/2) + np.around(radius*math.sin(ind*math.pi/vector_length)))
            x2 = int((img_channel.shape[0]/2) + np.around(radius*math.cos(ind*math.pi/vector_length+math.pi)))
            y2 = int((img_channel.shape[0]/2) + np.around(radius*math.sin(ind*math.pi/vector_length+math.pi)))

            # Mirroring the circular-shaped vector
            # Without this, the vector would only be half of a circle
            for ind_m_x in range(3):
                for ind_m_y in range(3):
                    mask_up[ind_m_x, ind_m_y] = img_channel[(x1 - 1 + ind_m_x),  (y1 - 1 + ind_m_y)]
                    mask_down[ind_m_x, ind_m_y] = img_channel[(x2 - 1 + ind_m_x), (y2 - 1 + ind_m_y)]

            # Placing the mask using the secret key (seed)
            mark_mask[x1, y1] = data_mark[ind] * np.mean(mask_up)
            mark_mask[x2, y2] = data_mark[ind] * np.mean(mask_down)

        # Applying the additive mask, controlled by the implementation strength
        magnitude_m = magnitude + implementation_strength*mark_mask

        # Transforming the image back to spatial domain
        img_channel_marked = self.image_to_spatial(magnitude_m, phase)

        #TODO
        if np.amax(img_channel) > 1:
            img_channel_marked = img_channel_marked / np.amax(img_channel_marked)

        # Return image as uint8
        return skimage.img_as_ubyte(img_channel_marked)

    # TODO
    # 8. Apply GCR Masking
    # def gcr_masking(img_marked, gcr_method, color_profile):
    #    return img_masked

    # 9. Merge blocks
    @staticmethod
    def image_merge_blocks(img_masked, block_number):

        block_height = int(img_masked.shape[0])
        block_width = int(img_masked.shape[1])
        # block_depth = int(block_number ** 2)
        image_height = int(block_height * block_number)
        image_width = int(block_width * block_number)

        img_merged = np.zeros((image_height, image_width))

        depth_counter  = 0
        x = 0
        y = 0

        while y < image_height:
            while x < image_width:
                img_merged[y : (y + block_height), 
                            x : (x + block_width)] = img_masked[:, :, depth_counter]
                x += block_width
                depth_counter += 1
            y += block_height
            x = 0

        return img_merged

    # 10. Merge marked K channel with original C M Y
    @staticmethod
    def image_merge_channels(img_orig, img_merged):

        # Checking if image is grayscale or color
        if len(img_orig.shape) == 2:
            return img_merged
        elif len(img_orig.shape) == 3:
            # Checking if color image is RGB or CMYK
            if img_orig.shape[-1] == 3:
                img_merged = np.dstack((img_merged, img_orig[:, :, 1:]))
                return img_merged
            elif img_orig.shape[-1] == 4:
                img_merged = np.dstack((img_orig[:, :, 0:3], img_merged))
                return img_merged
        else:
            raise AttributeError("Unknown image format.")

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
    def decode_data(img, length, frequency):

        if not img.shape == (512, 512):
            img = trs.resize(img, (512, 512))

        magnitude, phase = stego_block.image_to_fourier(img)
        mark = stego_block.key_generator(length)
        radius = stego_block.vector_radius(img, frequency)

        value_array = np.zeros(64)
        counter = 0

        for ind in range(int(radius - 32), int(radius + 32)):
            vec = stego_block.mark_extract(magnitude, ind)
            mark_reshaped = stego_block.mark_generate(mark, magnitude, ind)
            value_array[counter] = stego_block.mark_corr(mark_reshaped, vec)
            counter += 1

        return value_array

    # 7. Grubbs' test
    def grubbs_test(decode_values, alpha):
        return grubbs_values

    # 8. Extract message (matched pattern)
    def pattern_matching(grubbs_values):
        return pattern_output

    ### BRIDGE METHODS FOR EMBEDDING ###
    ### BRIDGE METHODS FOR EMBEDDING ###
    ### BRIDGE METHODS FOR EMBEDDING ###

    def vector_radius(img, frequency):

        if frequency == 'LOW':
            radius = 1/8 * (img.shape[0])
        elif frequency == 'MEDIUM':
            radius = 1/4 * (img.shape[0])
        elif frequency == 'HIGH':
            radius = 3/8 * (img.shape[0])
        else:
            raise AttributeError("Unknown frequency. Please use LOW, MEDIUM or HIGH")
        return radius

    def key_generator(length):

        key = np.random.randint(2, size = (length, 1)).astype(np.float32)
        return key

    def image_to_fourier(img):

        skimage.img_as_float64(img)
        fft2 = fftpack.fft2(img)
        magnitude = fftpack.fftshift(np.absolute(fft2))
        phase = np.angle(fft2)
        return magnitude, phase

    def image_to_spatial(magnitude, phase):

        img_spatial = fftpack.ifft2(np.multiply(fftpack.ifftshift(magnitude), np.exp(1j * phase)))
        img_spatial = np.real(img_spatial)
        return img_spatial
    
    ### BRIDGE METHOD S FOR DECODING ###
    ### BRIDGE METHOD S FOR DECODING ###
    ### BRIDGE METHOD S FOR DECODING ###

    def mark_extract(img, radius):

        step = math.pi / (2 * math.asin(1 / (2*radius)))
        vec = np.zeros((math.ceil(step), 1))
        mask = np.zeros((3, 3))

        for ind in range(math.ceil(step)):
            x1 = int((img.shape[0] / 2) +
                np.around(radius * math.cos(ind * math.pi/step)))
            y1 = int((img.shape[0] / 2) +
                np.around(radius * math.sin(ind * math.pi/step)))

            for  ind_m_x in range(3):
                for ind_m_y in range(3):
                    mask[ind_m_x, ind_m_y] = img[(x1-1 + ind_m_x), (y1-1 + ind_m_y)]
            vec[ind, 0] = np.amax(mask)
        return vec

    def mark_generate(mark, img, radius = 128):

        mask = np.zeros(img.shape)
        length = len(mark)

        for ind in range(length):
            x1 = int((img.shape[0] / 2) +
                np.around(radius * math.cos(ind * math.pi/length)))
            y1 = int((img.shape[0] / 2) +
                np.around(radius * math.sin(ind * math.pi/length)))
            
            mask[x1, y1] = mark[ind]
        return stego_block.mark_extract(mask, radius)

    def mark_corr(mark, vector):

        mark = mark - np.mean(mark)
        vector = vector - np.mean(vector)

        vector_length = len(vector)
        max_corr = np.zeros(vector_length)
        counter = 0

        for i in range(len(vector)):
            vector_roll = np.roll(vector, counter)
            vector_2d = np.reshape(vector_roll, (vector_length, 1))
            max_corr[counter] = np.corrcoef(mark[:, 0], vector_2d[:, 0])[0][1]
            counter += 1
        return np.amax(max_corr)