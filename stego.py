### Stego project from scratch ###

from os import stat 
import numpy as np
from PIL import Image
from scipy import fftpack
import skimage
import skimage.metrics as msr
import skimage.color as color
import skimage.transform as trs
import math
import cv2
from outliers import smirnov_grubbs as grubbs
from skimage.util import dtype

### Well, not exactly from scratch ###
# from wmark import WaterMark

# TODO
# Seed generator for permutation of blocks
# Encoder embeds data into first X permuted blocks
# Encoder has a list of possible patterns
# Decoder does not need to know the order of embedding
# Decoder only needs to know possible patterns
# Decoder checks each block for each possible pattern (until the pattern is found)

class stego_block:

    def __init__(self, seed_pattern, seed_message, seed_permutation):
        self.seed_pattern = seed_pattern
        self.seed_message = seed_message
        self.seed_permutation = seed_permutation
        pass

    def generate_pattern(self, length):

        # Using the self object, generating a seed
        np.random.seed(self.seed_pattern)
        pattern_a = np.random.randint(2, size = (length, 1)).astype(np.float32)
        pattern_b = np.random.randint(2, size = (length, 1)).astype(np.float32)

        return pattern_a, pattern_b

    def generate_message(self, bit_amount):

        np.random.seed(self.seed_message)
        message = np.random.randint(2, size = (bit_amount, 1))

        letter_message = []
        for i in range(bit_amount):

            if message[i] == 0:
                letter_message.append('A')
            elif message[i] == 1:
                letter_message.append('B')
            else:
                raise AttributeError("Wrong message input.")

        return letter_message

    def generate_permutation(self, block_number):

        blocks_ordered = np.arange(block_number)

        np.random.seed(self.seed_permutation)
        return np.random.permutation(blocks_ordered)

    ### IMAGE PROCESSING ###
    ### IMAGE PROCESSING ###
    ### IMAGE PROCESSING ###

    @staticmethod
    def image_read(filepath):

        # Read image as Numpy array
        img_orig = np.array(Image.open(filepath))
        return img_orig

    @staticmethod
    def image_resize(img, dimension):

        # Resize image to a dimension of choice
        if not img.shape == (dimension, dimension):
            # Preserve range in order to keep the same format
            img = trs.resize(img, (dimension, dimension), preserve_range = True)
        return img

    @staticmethod
    def image_save(img, filename, format, print_statement = 'NO'):
        img_object = Image.fromarray(img, str(format))
        img_object.save(str(filename))
        if print_statement == 'NO':
            pass
        elif print_statement == 'YES':
            print(f"Saved {img.name} image {img.shape}")
        else:
            pass
            raise Warning("Please provide a YES or NO answer for print statement.")

    @staticmethod
    def extract_channel(img_orig):

        # Checking if image is grayscale or color
        if len(img_orig.shape) == 2:
            return img_orig
        elif len(img_orig.shape) == 3:
            # Checking if color image is CMYK or RGB
            if img_orig.shape[-1] == 3:
                img_orig = color.rgb2ycbcr(img_orig)
                return img_orig[:, :, 0]
            elif img_orig.shape[-1] == 4:
                return img_orig[:, :, -1]
        else:
            raise AttributeError("Please provide grayscale, RGB or CMYK image.")

    @staticmethod
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

    ### ENCODER ###
    ### ENCODER ###
    ### ENCODER ###

    # TODO
    def activity_test(img_channel_blocks, activity_threshold):
        return activity_map

    def implementation_strength(self, message, img_block, psnr_range, length, frequency):

        implementation_limits = (50, 150000)

        l = implementation_limits[0]
        r = implementation_limits[1]

        counter = 0

        while r >= l:
            counter += 1
            if counter >= 25:
                break
            mid = l + (r - l) / 2

            implementation_strength = mid

            img_marked = self.embed_data(
                message = message, 
                img_channel = img_block, 
                length = length, 
                frequency = frequency, 
                factor = implementation_strength
            )
            img_zero = self.embed_data(
                message = message, 
                img_channel = img_block, 
                length = length, 
                frequency = frequency, 
                factor = 0
            )

            psnr_value = msr.peak_signal_noise_ratio(img_zero, img_marked)

            if psnr_value > psnr_range[1]:
                l = mid
            elif psnr_value < psnr_range[0]:
                r = mid 
            else:
                return implementation_strength, psnr_value

        # else:
        print("Error: Unable to find. Set Impact Factor to 1000.")
        print(psnr_value)
        return [1000, psnr_value]

    def embed_data(self, message, img_channel, length, mask, frequency, factor):

        # Get radius from provided frequency range
        radius = stego_block.vector_radius(img_channel, frequency)

        # Generate mark using the secret key (seed)
        if message == 'A':
            data_mark = self.generate_pattern(length)[0]
        elif message == 'B':
            data_mark = self.generate_pattern(length)[1]
        else:
            raise AttributeError("Unknown message. Encoder is confused.")

        # Transform the input image into the frequency domain
        magnitude, phase = self.image_to_fourier(img_channel)

        # Define the data mask (area where the data will be embedded)
        mark_mask = np.zeros(img_channel.shape)
        mask_shape = mask
        mask_up = np.zeros(mask_shape)
        mask_down = np.zeros(mask_shape)

        # Defining key points on the circular-shaped vector
        for ind in range(length):
            # x1 = int((img_channel.shape[0]/2) + np.around(radius*math.cos(ind*math.pi/length)))
            # y1 = int((img_channel.shape[0]/2) + np.around(radius*math.sin(ind*math.pi/length)))
            x1 = int((img_channel.shape[0]/2) + np.around(radius*math.cos(ind*math.pi/length + math.pi/8)))
            y1 = int((img_channel.shape[0]/2) + np.around(radius*math.sin(ind*math.pi/length + math.pi/8)))
            
            # x2 = int((img_channel.shape[0]/2) + np.around(radius*math.cos(ind*math.pi/length+math.pi)))
            # y2 = int((img_channel.shape[0]/2) + np.around(radius*math.sin(ind*math.pi/length+math.pi)))
            x2 = int((img_channel.shape[0]/2) + np.around(radius*math.cos(ind*math.pi/length+math.pi + math.pi/8)))
            y2 = int((img_channel.shape[0]/2) + np.around(radius*math.sin(ind*math.pi/length+math.pi + math.pi/8)))

            # Mirroring the circular-shaped vector
            # Without this, the vector would only be half of a circle
            for ind_m_x in range(mask[0]):
                for ind_m_y in range(mask[0]):
                    mask_up[ind_m_x, ind_m_y] = magnitude[(x1 - 1 + ind_m_x),  (y1 - 1 + ind_m_y)]
                    mask_down[ind_m_x, ind_m_y] = magnitude[(x2 - 1 + ind_m_x), (y2 - 1 + ind_m_y)]

            # Placing the mask using the secret key (seed)
            mark_mask[x1, y1] = data_mark[ind] * np.mean(mask_up)
            mark_mask[x2, y2] = data_mark[ind] * np.mean(mask_down)

        # Applying the additive mask, controlled by the implementation strength
        magnitude_m = magnitude + factor*mark_mask

        # Saving image
        # magnitude_log = stego_block.frequency_domain_log(magnitude)
        # stego_block.image_save(magnitude_log, 'test_set/magnitude_log.jpg', 'L')

        # Saving image
        # magnitude_m_log = stego_block.frequency_domain_log(magnitude_m)
        # stego_block.image_save(magnitude_m_log, 'test_set/magnitude_m_log.jpg', 'L')

        # Transforming the image back to spatial domain
        img_channel_marked = self.image_to_spatial(magnitude_m, phase)

        #TODO
        if np.amax(img_channel) > 1:
            img_channel_marked = img_channel_marked / np.amax(img_channel_marked)

        # Return image as uint8
        return skimage.img_as_ubyte(img_channel_marked), magnitude[x1, y1], magnitude_m[x1, y1]

    def embed_pattern_to_blocks(self, message, permutation, image_blocks, length, frequency):

        blocks_marked = np.copy(image_blocks)
        
        for i in range(len(message)):

            factor = self.implementation_strength(
                message = message[i],
                img_block = image_blocks[:, :, permutation[i]],
                psnr_range = (25, 30),
                length = 200,
                frequency = 'MEDIUM'
            )

            blocks_marked[:, :, permutation[i]] = self.embed_data(
                message = message[i],
                img_channel = image_blocks[:, :, permutation[i]],
                length = length,
                frequency = frequency,
                factor = factor[0]
                # factor = 15000
            )
            print(f"Embedding message {message[i]} in block {permutation[i]}, factor = {factor[0]}, PSNR = {round(factor[1], 3)}")
            # print(f"Embedding message {message[i]} in block {permutation[i]}.")

        return blocks_marked

    # TODO
    def gcr_masking(img_marked, gcr_method, color_profile):
        return img_masked

    ### FREQUENCY DOMAIN ###
    ### FREQUENCY DOMAIN ###
    ### FREQUENCY DOMAIN ###

    @staticmethod
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
        
    @staticmethod
    def image_to_fourier(img):

        # Transforming image to float
        skimage.img_as_float64(img)
        
        # Transforming image from spatial to frequency domain
        # Fourier Transform is complex, creating both magnitude and phase
        fft2 = fftpack.fft2(img)
        magnitude = fftpack.fftshift(np.absolute(fft2))
        phase = np.angle(fft2)

        # Return magnitude and phase as separate objects
        return magnitude, phase

    @staticmethod
    def frequency_domain_log(magnitude):

        # Log of original magnitude
        m_log = 255 / np.log(1 + np.max(magnitude))
        magnitude_log = m_log * (np.log(magnitude + 1))
        magnitude_log = np.array(magnitude_log, dtype = np.uint8)

        return magnitude_log

    @staticmethod
    def image_to_spatial(magnitude, phase):

        # Using the magnitude and phase to return image to spatial domain
        img_spatial = fftpack.ifft2(np.multiply(fftpack.ifftshift(magnitude), np.exp(1j * phase)))
        img_spatial = np.real(img_spatial)
        return img_spatial

    ### VECTORS ###
    ### VECTORS ###
    ### VECTORS ###
    # TODO
    # Should these be static methods?
    # Or are they a part of the decoder?

    @staticmethod 
    def mark_extract(img, radius, mask = (3, 3)):

        step = math.pi / (2 * math.asin(1 / (2*radius)))
        vec = np.zeros((math.ceil(step), 1))
        mask = np.zeros(mask)

        for ind in range(math.ceil(step)):
            x1 = int((img.shape[0] / 2) +
                np.around(radius * math.cos(ind * math.pi/step)))
            y1 = int((img.shape[0] / 2) +
                np.around(radius * math.sin(ind * math.pi/step)))

            for  ind_m_x in range(3):
                for ind_m_y in range(3):
                    mask[ind_m_x, ind_m_y] = img[(
                        x1-1 + ind_m_x), (y1-1 + ind_m_y)]
            vec[ind, 0] = np.amax(mask)
        return vec

    @staticmethod
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

    @staticmethod
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

    ### DECODER ###
    ### DECODER ###
    ### DECODER ###

    # 1. Read cmyk image
    # 2. Split image into blocks
    # 3. Choose K channels
    # 4. Activity test
    # 5. Apply Fourier transform
    # 6. Search for watermarks

    def grubbs_test(self, pattern, img_block, length, frequency, alpha):

        values_check = self.detect_pattern(pattern, img_block, length, frequency)
        values_grubbs = grubbs.max_test_outliers(values_check, alpha = alpha)

        if len(values_grubbs) == 0:
            return False
        elif len(values_grubbs) != 0:
            return True
        else:
            raise AttributeError("Grubbs' test failed. Please try again.")

    def detect_pattern(self, pattern, img, length, frequency):

        magnitude, phase = self.image_to_fourier(img)

        if pattern == 'A':
            mark = self.generate_pattern(length)[0]
        elif pattern == 'B':
            mark = self.generate_pattern(length)[1]
        else:
            raise AttributeError("Please provide 'A' or 'B' as pattern choice.")

        radius = stego_block.vector_radius(img, frequency)

        value_array = np.zeros(64)
        counter = 0

        for ind in range(int(radius - 32), int(radius + 32)):
            vec = stego_block.mark_extract(magnitude, ind)
            mark_reshaped = stego_block.mark_generate(mark, magnitude, ind)
            value_array[counter] = stego_block.mark_corr(mark_reshaped, vec)
            counter += 1

        return value_array

    def decode_data_pattern(self, permutation, image_blocks, length, frequency, alpha):

        decoded_values = np.zeros((image_blocks.shape[-1], 1))
        decoded_message = []
        
        for i in range(image_blocks.shape[-1]):

            decoded_values[[i]] = self.grubbs_test(
                pattern = 'A',
                img_block = image_blocks[:, :, permutation[i]],
                length = length,
                frequency = frequency,
                alpha = alpha
            )

            if decoded_values[i] == 1:

                print(f"Found pattern A in block {permutation[i]}.")
                decoded_message.append('A')

            elif decoded_values[i] == 0:

                print(f"No pattern A in block {permutation[i]}.")

                decoded_values[i] = self.grubbs_test(
                    pattern = 'B',
                    img_block = image_blocks[:, :, permutation[i]],
                    length = length,
                    frequency = frequency,
                    alpha = alpha
                )

                if decoded_values[i] == 1:

                    print(f"Found pattern B in block {permutation[i]}")
                    decoded_message.append('B')

                elif decoded_values[i] == 0:

                    print(f"No pattern B in block {permutation[i]}")

        decoded_values = decoded_values.reshape(
            (int(math.sqrt(decoded_values.shape[0])), 
            int(math.sqrt(decoded_values.shape[0])))
        )

        return decoded_values, decoded_message