import math
import warnings 

import numpy as np
from PIL import Image
import PIL
from skimage import color, transform, data, restoration
from skimage.filters import unsharp_mask
import cv2
from scipy.spatial.distance import pdist, squareform
from scipy.signal import convolve2d
import imageio

from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

import glob
import os
from pathlib import Path

# import dlib

from wmark import WaterMark

# Define new row
nl = '\n' 

class bitmapTransform:

    @staticmethod
    def readBitmap(path):
    
        bitmapRead = imageio.imread(path)
        return bitmapRead

    @staticmethod
    def bitmapFlatten(bitmap2D):

        bitmap1D = bitmap2D.flatten('C')
        return bitmap1D

    @staticmethod
    def bitmapReturn(bitmap1D):

        dimensions2D = int(np.sqrt(bitmap1D.shape[0]))
        bitmap2D = bitmap1D.reshape((dimensions2D, dimensions2D))
        return bitmap2D

class blockProc:

    ### OLD ###
    ### OLD ###
    ### OLD ###

    @staticmethod
    def blockDivide(img, blocksWidth, blocksHeight):
        """Divides input image into {blocksWidth} * {blocksHeight} blocks.

        Args:
            img (string): Path to input image.
            blocksWidth (int): Desired number of blocks per axis.
            blocksHeight (int): Desired number of blocks per axis.

        Raises:
            AttributeError: If axis dimension is not divisible with {blocksWidth}.
            AttributeError: If axis dimension is not divisible with {blocksHeight}.

        Returns:
            [ndarray]: Array of blocks, with third axis representing the number of blocks.
        """

        # Read image
        # imgArray = np.array(Image.open(img))
        imgArray = img

        # Warning if image is not grayscale and transform to grayscale
        if len(imgArray.shape) != 2:
            warnings.warn("Method will transform input image into grayscale.")
            imgArray = color.rgb2ycbcr(imgArray)
            imgArray = imgArray[:, :, 0]

        # Set output array depth
        arraySizeZ = blocksWidth * blocksHeight
        
        # Size of each block
        arraySizeX = int(imgArray.shape[1] / blocksWidth)
        arraySizeY = int(imgArray.shape[0] / blocksHeight)

        if imgArray.shape[0] % blocksHeight != 0:
            raise AttributeError(
        f"Can not compute ({imgArray.shape[0]} / {blocksHeight}). Please provide a number of blocks divisible with original height.")

        if imgArray.shape[1] % blocksWidth != 0:
            raise AttributeError(
        f"Can not compute ({imgArray.shape[1]} / {blocksWidth}). Please provide a number of blocks divisible with original height.")

        # Shape of numpy array to be filled
        outputBlocks = np.zeros((arraySizeY, arraySizeX, arraySizeZ))

        # Split the Y channel of the image into vertical blocks
        split_a = np.split(imgArray, blocksHeight, axis = 0)

        # Set counter to zero
        counter = 0

        for i in range(blocksHeight):
            for j in range(blocksWidth):

                # Split vertical blocks into square blocks
                split_b = np.split(split_a[i], blocksWidth, axis = 1)

                # Fill Numpy array with blocks
                outputBlocks[:, :, counter] = split_b[j]

                # Increase counter
                counter += 1

        return outputBlocks

    @staticmethod
    def blockResize(blocks, size):

        # Defining number of blocks
        blockNumber = blocks.shape[2]

        # Creating array to be filled with embedded block data
        blockResized = np.zeros((size, size, blockNumber))

        i = 0

        while i < blockNumber:
            blockResized[:, :, i] = WaterMark.imresize(img = blocks[:, :, i], outputDim = size)

            i += 1

        return blockResized 

    @staticmethod
    def blockMerge(array, imgWidth, imgHeight):
        """Merges image blocks into a grayscale image.

        Args:
            array (ndarray): Containing image blocks with third axis representing the number of blocks.
            imgWidth (int): Width of full image.
            imgHeight (int): Height of full image.

        Returns:
            (ndarray): Array containing full grayscale image.
        """

        # Dimension of the input image
        blockSizeX = array.shape[1]
        blockSizeY = array.shape[0]

        # Number of blocks along an axis
        blocksAlongX = imgWidth / blockSizeX
        blocksAlongY = imgHeight / blockSizeY

        # Creating Numpy Array for output image
        outputImage = np.zeros((imgHeight, imgWidth))

        # Counter for indexing each block from input array
        depthCounter = 0

        # Starting position of X
        x = 0

        # Starting position of Y
        y = 0

        while y < imgHeight:
            while x < imgWidth:
                outputImage[y : (y + blockSizeY), 
                            x : (x + blockSizeX)] = array[:, :, depthCounter]
                x += blockSizeX
                depthCounter += 1

            y += blockSizeY
            x = 0

        return outputImage

    ### NEW ###
    ### NEW ###
    ### NEW ###

    def imgRead(img):
    
        # Read image
        image = np.array(Image.open(img))
        if image.shape[-1] == 3:
            return color.rgb2ycbcr(image).astype('uint8')
        elif image.shape[-1] == 4:
            return image
        elif len(image.shape) == 2:
            return image
        else:
            raise ValueError("Not a valid image format.")

    def imgPreprocess(img):
        if img.shape[-1] == 4:
            gray = cv2.bilateralFilter(img[:, :, -1], 11, 17, 17)
        elif len(img.shape) == 2:
            gray = cv2.bilateralFilter(img, 11, 17, 17)
        else:
            raise ValueError("Not a valid image format.")

        edged = cv2.Canny(gray, 30, 200)

        return edged

    @staticmethod
    def imgSharpen(img, method):

        if method == 'sharp':
            imgSharp = unsharp_mask(img, radius = 5.0, amount = 2.0, preserve_range = True)
            return imgSharp
        elif method == 'wiener':
            psf = np.ones((5, 5)) / 25
            img = convolve2d(img, psf, 'same')
            img += 0.1 * img.std() * np.random.standard_normal(img.shape)
            imgWiener = restoration.wiener(img, psf, 1100)
            return imgWiener
        else:
            raise AttributeError('Please select a proper method.')

    @staticmethod
    def getImageChannels(img):

        # Divide image into channels
        # img_ycbcr = color.rgb2ycbcr(img).astype('uint8')

        return img[:, :, 0], img[:, :, 1], img[:, :, 2]

    @staticmethod
    def mergeImageChannels(img_y, img_cb, img_cr):

        img_ycbcr = np.dstack((img_y, img_cb, img_cr))

        # Merge channels into RGB image
        # img_rgb = color.ycbcr2rgb(img_ycbcr)
        # img_rgb = (img_rgb * 255)

        return img_ycbcr

    @staticmethod
    def mergeCMYK(imgC, imgM, imgY, imgK):

        img_cmyk = np.dstack((imgC, imgM, imgY, imgK))
        return img_cmyk

    @staticmethod
    def imgCrop(img, blockSize):

        # Calculate half of block size
        blockRadius = int(blockSize / 2)

        # Define cropped image
        imgCrop = img[(blockRadius):(img.shape[0] - blockRadius), 
                      (blockRadius):(img.shape[1] - blockRadius)]

        # Now we are certain the blocks will not 
        # reach out of bounds of original image
        return imgCrop
    
    @staticmethod
    def findCenters(imgCrop, centerNumber, blurKsize = 10):

        # Define blur ksize 
        ksize = (blurKsize, blurKsize) 
  
        # Blur image
        imageBlurred = cv2.blur(imgCrop, ksize)
        # blockProc.saveImage(imageBlurred)
    
        # Find feature points in cropped & blurred image
        centers = cv2.goodFeaturesToTrack(imageBlurred, centerNumber, 0.01, 10)
        centers = np.int0(centers)

        # For some reason, this array output is clearer
        centersArray = np.asarray(centers)

        return centersArray[:, 0, :]

    @staticmethod
    def findFaceFeatures(img, feat1, feat2, feat3, feat4):

        featurePoints = np.zeros((4, 2))

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        img = cv2.imread(img)
        gray = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
    
            landmarks = predictor(image = gray, box = face)
    
            featurePoints[0, 0] = landmarks.part(feat1).x
            featurePoints[0, 1] = landmarks.part(feat1).y
            
            featurePoints[1, 0] = landmarks.part(feat2).x
            featurePoints[1, 1] = landmarks.part(feat2).y
            
            featurePoints[2, 0] = landmarks.part(feat3).x
            featurePoints[2, 1] = landmarks.part(feat3).y
            
            featurePoints[3, 0] = landmarks.part(feat4).x
            featurePoints[3, 1] = landmarks.part(feat4).y

        return featurePoints

    @staticmethod 
    def translateCenters(centers, img, imgCrop):

        # Define block radius
        blockRadius = (abs(img.shape[0] - imgCrop.shape[0])) / 2

        # Adding blockRadius values to points, 
        # in order to match oordinates on original image
        centersModified = [x + blockRadius for x in centers]

        # For some reason, this array output is clearer
        centersArray = np.asarray(centersModified)

        return centersArray

    @staticmethod
    def extractBlock(img, centers, blockSize):
        
        # Define number of centers and block radius
        centerNumber = centers.shape[0]
        print(f"Center number: {centerNumber}")
        blockRadius = blockSize / 2
        print(f"Block Radius: {blockRadius}")

        # Create empty array to be filled with image date
        imgBlockArray = np.zeros((blockSize, blockSize, centerNumber))

        i = 0

        while i < centerNumber:
    
            # Defining starting and end points of block
            centerX1 = int(centers[i, 0] - blockRadius)
            centerX2 = int(centers[i, 0] + blockRadius)
            centerY1 = int(centers[i, 1] - blockRadius)
            centerY2 = int(centers[i, 1] + blockRadius)
            
            print(f"Extracting {blockSize} px block from center {centers[i]}...")

            # Filling array with image data
            imgBlockArray[:, :, i] = img[centerY1 : centerY2, centerX1 : centerX2]

            i += 1

        return imgBlockArray

    @staticmethod
    def embedBlock(blocks, length, impactFactor, seedNumber):

        # Defining number of blocks
        blockNumber = blocks.shape[2]

        # Creating array to be filled with embedded block data
        blockEmbed = np.zeros_like(blocks)

        i = 0

        while i < blockNumber:
            wObject = WaterMark(seedNumber)

            # impactFactor = wObject.findImpactFactor(img = blocks[:, :, i], rangePSNR = (20, 30))

            # Filling array with embedded block data
            blockEmbed[:, :, i] = wObject.embedMark(img = blocks[:, :, i], length = length, factor = impactFactor)

            # print(f"Embedding watermark with seed No. {seedNumber} with impact factor {impactFactor}...")
            
            i += 1

        return blockEmbed

    @staticmethod
    def decodeBlockRaw (blocks, seedNumber, length):

        # Defining number of blocks
        blockNumber = blocks.shape[2]

        blockDecode = np.zeros((blockNumber))

        i = 0

        while i < blockNumber:
            wObject = WaterMark(seedNumber)

            # blockDecode[i, 0] = wObject.decodeMark(img = blocks[:, :, i], length = length, metric = 'CORR')
            blockDecode[i], temp = wObject.detectOutlier(img = blocks[:, :, i], metric = 'CORR', length = length, alpha = 0.001)

            i += 1

        return blockDecode

    @staticmethod
    def returnBlock(img, centers, blocks):

        # Defining number of blocks and block radius
        blockNumber = blocks.shape[2]
        blockRadius = blocks.shape[0] / 2

        i = 0

        while i < blockNumber:
    
            # Defining starting and end points of block
            centerX1 = int(centers[i, 0] - blockRadius)
            centerX2 = int(centers[i, 0] + blockRadius)
            centerY1 = int(centers[i, 1] - blockRadius)
            centerY2 = int(centers[i, 1] + blockRadius)

            # Returning embedded block into original image
            img[centerY1 : centerY2, centerX1 : centerX2] = blocks[:, :, i]

            print(f"Returned center No. {i} at {centers[i]}...")
            i += 1

        return img

    @staticmethod
    def decodeBlock(img, centers, blockSize, seedNumber):

        # Defining number of blocks, block radius
        blockNumber = centers.shape[0]
        blockRadius = blockSize / 2

        # Creating array to be filled with decode values
        blockDecode = np.zeros((blockNumber))

        i = 0

        while i < blockNumber:
        
            # Defining starting and end points of block
            centerX1 = int(centers[i, 0] - blockRadius)
            centerX2 = int(centers[i, 0] + blockRadius)
            centerY1 = int(centers[i, 1] - blockRadius)
            centerY2 = int(centers[i, 1] + blockRadius)

            wObject = WaterMark(seedNumber)

            print(f"Decoding {blockSize} px block with seed No. {seedNumber} from center {centers[i]}...")

            # Filling array with decode value
            blockDecode[i] = wObject.decodeMark(img[centerY1 : centerY2, centerX1 : centerX2], metric = 'CORR')

            i += 1

        return blockDecode

    @staticmethod
    def StackOverflowFilterCenters(centers, blockSize):
    
        # Creating distance matrix; chebyshev is used for square blocks
        distances = squareform(pdist(centers, metric = 'chebyshev'))

        # Creating array of centers that have not been dropped (starting with all centers)
        indices = np.arange(centers.shape[0])

        # Making sure first index is always included
        out = [0]

        while True:
            try:
                # Drop centers inside threshold 
                indices = indices[distances[indices[0], indices] > blockSize]

                # Add next index that has not been dropped to the output
                out.append(indices[0])
    
            except:
                # Once out of centers, IndexError for stopping
                break

        centersFiltered = centers[out]

        return centersFiltered

    @staticmethod
    def saveBlock(array):

        # Source Directory
        src_folder = os.getcwd()
        # Source Path
        src_pth = Path(src_folder).resolve()

        # Creating folder for blocks
        dst_folder = Path(src_folder+'/savedBlock/').resolve()
        Path(dst_folder).mkdir(exist_ok = True)

        # Saving blocks
        for i in range(array.shape[2]):
            imgObject = Image.fromarray(array[:, :, i].astype('uint8'), 'L')
            imgName = f"{str(dst_folder)}/block_{i}.jpg"
            imgObject.save(imgName)

    @staticmethod
    def saveImage(array, filename, imgMode, extension):

        # Source Directory
        src_folder = os.getcwd()
        # Source Path
        src_pth = Path(src_folder).resolve()
    
        # Creating folder for blocks
        dst_folder = Path(src_folder+'/savedImage/').resolve()
        Path(dst_folder).mkdir(exist_ok = True)

        # Saving image
        imgObject = Image.fromarray(array[:, :].astype('uint8'), imgMode)
        imgName = f"{filename}.{extension}"
        imgObject.save(imgName)

    @staticmethod
    def identicalTest(centersOrig, centersEmbed):

        return np.array_equal(centersOrig, centersEmbed)

    @staticmethod
    def wienerFilter(img, noiseAmount):

        if len(img.shape) == 2:
            img_unfiltered = img / 255
        elif len(img.shape) == 3: 
            img_unfiltered = img[:, :, -1] / 255

        psf = np.ones((5, 5)) / 25

        img_unfiltered = conv2(img_unfiltered, psf, 'same')
        img_unfiltered += noiseAmount * img_unfiltered.std() * np.random.standard_normal(img_unfiltered.shape)

        deconvolved, _ = restoration.unsupervised_wiener(img_unfiltered, psf)

        return deconvolved