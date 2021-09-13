#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Ante Poljicak, Petar Branislav Jelusic
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.


import skimage
import skimage.metrics as msr
import skimage.color as color
import skimage.transform as trs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
from PIL import Image, ImageCms
from scipy import fftpack
import glob
import math
import multiprocessing
from multiprocessing import Queue as PQueue
import queue
import pandas as pd
import wmgcr
from outliers import smirnov_grubbs as grubbs
import gcrpywrap as gw
import ctypes
from pathlib import Path


class WaterMark:

    # instance atributes

    # constructor with seed
    def __init__(self, seed):
        self.seed = seed
        pass

    @staticmethod
    def compareImages(originalImg, modifiedImg, maskedImg, profilePath):
        """ Helper method for displaying comparison of images """
        fig, axes = plt.subplots(nrows=1, ncols=3, sharex='all', sharey='all')

        label = 'PSNR: {:.2f}'

        # check if the image is grayscale, RGB or CMYK image
        if len(originalImg.shape) == 2:
            displayOriginalImg = originalImg
            displayModifiedImg = modifiedImg
            displayMaskedImg = maskedImg
        elif len(originalImg.shape) == 3:
            if originalImg.shape[2] == 4:  # CMYK image
                print('Unable to show images correctly in cmyk > coverting to srgb')
                displayOriginalImg = WaterMark.profileCmyk2srgb(
                    originalImg, profileCmyk=profilePath)
                displayModifiedImg = WaterMark.profileCmyk2srgb(
                    modifiedImg, profileCmyk=profilePath)
                displayMaskedImg = WaterMark.profileCmyk2srgb(
                    maskedImg, profileCmyk=profilePath)

            elif originalImg.shape[2] == 3:  # RBG image
                displayOriginalImg = originalImg
                displayModifiedImg = modifiedImg
                displayMaskedImg = maskedImg

            else:
                raise TypeError(
                    "Image has wrong number of channels. Only 3 or 4 channels allowed")
        else:
            raise TypeError(
                "Image isn't of correct type. Only grayscale, RGB and CMYK allowed")

        axes[0].imshow(displayOriginalImg)
        axes[0].set_title('Original image')

        axes[1].imshow(displayModifiedImg)
        axes[1].set_title('Modified image')

        axes[2].imshow(displayMaskedImg)
        axes[2].set_title('Masked image')

        plt.show()

    # TODO Make readable regardless of image profile.
    @staticmethod
    def imread(imgName):
        """ read image and return np array """
        return np.array(Image.open(imgName))

    def inputProc(self, img):
        """ Takes image and returns its magnitude and phase in fourier domain"""
        # convert to float64 to keep precision
        skimage.img_as_float64(img)
        fft2 = fftpack.fft2(img)
        # shift lowest frequency to the center
        magnitude = fftpack.fftshift(np.absolute(fft2))
        phase = np.angle(fft2)
        return magnitude, phase

    def pseudoGen(self, length):
        """ Takes lenght and returns binary pseudorandom vector of that length """
        np.random.seed(self.seed)
        return np.random.randint(2, size=(length, 1)).astype(np.float32)

    def outputProc(self, magnitude, phase):
        """ Takes magnitude and phase of the image and returns img in spatial domain"""
        img = fftpack.ifft2(np.multiply(
            fftpack.ifftshift(magnitude), np.exp(1j * phase)))
        img = np.real(img)  # Image still is complex so take only real part
        return img

    @staticmethod
    def getMarkChannel(img):
        """getter for channel used for embeding watermark

        Arguments:
            img {ndarray} -- host image

        Raises:
            TypeError: raised if image has wrong number of channels

        Returns:
            ndarray -- channel in which to embed watermark
            str -- image type 'GRAY', 'CMYK', 'RGB'
        """
        # check if the image is grayscale, RGB or CMYK image
        if len(img.shape) == 2:
            image_type = 'GRAY'
            img_y = img
        elif len(img.shape) == 3:
            if img.shape[2] == 4:  # CMYK image
                image_type = 'CMYK'
                img_y = img[:, :, 3]
            elif img.shape[2] == 3:  # RBG image
                image_type = 'RGB'
                img_ycbcr = color.rgb2ycbcr(img)
                img_y = img_ycbcr[:, :, 0]
            else:
                raise TypeError(
                    "Image has wrong number of channels. Only 3 or 4 channels allowed")
        else:
            raise TypeError(
                "Image isn't of correct type. Only grayscale, RGB and CMYK allowed")

        return img_y, image_type

    def embedMark(self, img, length=200, frequencies='MEDIUM', factor=5):
        """ Method for embeding watermark in magnitude spectrum of the image

        Arguments:
            img {ndarray} -- host image

        Keyword Arguments:
            length {int} -- length of the watermark (default: {200})
            frequencies {str} -- frequencies in which watermark will be embedded (default: {'MEDIUM'})
            factor {int} -- implementation strength (default: {5})

        Returns:
            {ndarray} -- image with embedded watermark
        """

        img_y, image_type = WaterMark.getMarkChannel(img)

        radius = WaterMark.frequency2Radius(img, frequencies)

        # Generate prnd sequence
        mark = self.pseudoGen(length)

        # Transformation in Fourier domain
        magnitude, phase = self.inputProc(img_y)

        # Embed the vector into the image
        watermark_mask = np.zeros(img_y.shape)

        # 2 masks are needed to ensure simetric modification of magnitude spectrum
        mask_shape = (3, 3)
        mask_up = np.zeros(mask_shape)
        mask_down = np.zeros(mask_shape)

        for ind in range(length):
            x1 = int((img_y.shape[0]/2) +
                     np.around(radius*math.cos(ind*math.pi/length)))
            y1 = int((img_y.shape[0]/2) +
                     np.around(radius*math.sin(ind*math.pi/length)))
            x2 = int((img_y.shape[0]/2) + np.around(radius *
                                                    math.cos(ind*math.pi/length+math.pi)))
            y2 = int((img_y.shape[0]/2) + np.around(radius *
                                                    math.sin(ind*math.pi/length+math.pi)))
            for ind_m_x in range(3):
                for ind_m_y in range(3):
                    mask_up[ind_m_x, ind_m_y] = img_y[(
                        x1-1+ind_m_x), (y1-1+ind_m_y)]
                    mask_down[ind_m_x, ind_m_y] = img_y[(
                        x2-1+ind_m_x), (y2-1+ind_m_y)]

            # build watermark mask by multiplying mark bits
            # with mean value of the surrounding coefficients
            watermark_mask[x1, y1] = mark[ind]*np.mean(mask_up)
            watermark_mask[x2, y2] = mark[ind]*np.mean(mask_down)

        # add watermark mask multiplied by factor to the original magnitude
        # TODO modify code to beter use implementation factor maybe use mean magnitude of an image
        magnitude_m = magnitude+factor*watermark_mask

        # Transformation to Spatial domain
        img_y_marked = self.outputProc(magnitude_m, phase)

        if image_type == 'RGB':
            img_ycbcr = color.rgb2ycbcr(img)
            img_ycbcr_m = np.dstack((img_y_marked, img_ycbcr[:, :, 1:]))
            img_m = color.ycbcr2rgb(img_ycbcr_m)
        elif image_type == 'CMYK':
            img_m = np.dstack((img[:, :, 0:3], img_y_marked))
        else:
            img_m = img_y_marked

        # TODO this processing will change the data of unprocessed channels too. Figure out a way to fix that
        # check if image is in the -1 , 1 interval
        if np.amax(img_m) > 1:
            img_m = img_m/np.amax(img_m)

        # return image as uint8
        return skimage.img_as_ubyte(img_m)

    def decodeMark(self, img, metric, length=200, frequencies='MEDIUM'):
        """Method for decoding hidden watermark

        Arguments:
            img {ndarray} -- host image

        Keyword Arguments:
            metric {str} -- metric of choice for decoding values - covariation or correlation ('COV', 'CORR')
            length {int} -- length of embeded (watermark) 1D vector  (default: {200})
            frequencies {str} -- frequencies at which to search for vector ('LOW', 'MEDIUM', 'HIGH') (default: {'MEDIUM'})

        Returns:
            {float} -- covariation / correlation value of the extracted vector and generated watermark
        """
        metricArray = self.corrArray(img, metric, length, frequencies)
        return np.amax(metricArray)

    def corrArray(self, img, metric, length=200, frequencies='MEDIUM'):
        """Method for extracting entire array of possible decetion values

        Arguments:
            img {ndarray} -- host image

        Keyword Arguments:
            metric {str} -- metric of choice for decoding values - covariation or correlation ('COV', 'CORR')
            length {int} -- length of embeded (watermark) 1D vector  (default: {200})
            frequencies {str} -- frequencies at which to search for vector ('LOW', 'MEDIUM', 'HIGH') (default: {'MEDIUM'})

        Returns:
            {float} -- array of covariation / correlation values of the extracted vector and generated watermark
        """
        img_y, image_type = WaterMark.getMarkChannel(img)

        if not img_y.shape == (512, 512):
            img_y = trs.resize(img_y, (512, 512))

        magnitude, phase = self.inputProc(img_y)

        mark = self.pseudoGen(length)
        radius = WaterMark.frequency2Radius(img_y, frequencies)

        metricArray = np.zeros(64)
        counter = 0
        for ind in range(int(radius-32), int(radius+32)):
            vec = WaterMark.extractMark(magnitude, ind)
            reshapedMark = WaterMark.generateReshapedMark(mark, magnitude, ind)
            if metric == 'COV':
                metricArray[counter] = WaterMark.covarMark(reshapedMark, vec)
            elif metric == 'CORR':
                metricArray[counter] = WaterMark.corrMark(reshapedMark, vec)
            counter += 1

        return metricArray

    def detectOutlier(self, img, metric, alpha=0.0001):
        """Method for finding outliers in array of correlation values

        Arguments:
            img {ndarray} -- host image
            metric {str} -- metric of choice for decoding values - covariation or correlation ('COV', 'CORR')
            alpha {float} -- significance level used for grubbs' test
        Returns:
            {boolean} -- If at least one outlier is detected, returns True; If there are no outliers, returns False
        """
        PossibleOutliers = WaterMark.corrArray(self, img, metric)
        GrubbsDetecion = grubbs.max_test_outliers(
            PossibleOutliers, alpha=alpha)

        if len(GrubbsDetecion) == 0:
            return False
        elif len(GrubbsDetecion) != 0:
            return True
        else:
            raise AttributeError("Outlier test not valid. Please try again.")

    @staticmethod
    def extractMark(img, radius):
        """ Method for extracting vector from the ndarray given the radius

        Arguments:
            img {ndarray} -- host ndarray
            radius {int} -- radius at which to extract vector

        Returns:
            {ndarray} -- Extracted vector
        """
        # step between coefficients used for embedding.
        step = math.pi/(2*math.asin(1/(2*radius)))
        vec = np.zeros((math.ceil(step), 1))
        mask = np.zeros((3, 3))

        for ind in range(math.ceil(step)):
            x1 = int((img.shape[0]/2) +
                     np.around(radius*math.cos(ind*math.pi/step)))
            y1 = int((img.shape[0]/2) +
                     np.around(radius*math.sin(ind*math.pi/step)))

            for ind_m_x in range(3):
                for ind_m_y in range(3):
                    mask[ind_m_x, ind_m_y] = img[(
                        x1-1+ind_m_x), (y1-1+ind_m_y)]
            vec[ind, 0] = np.amax(mask)
        return vec

    @staticmethod
    def generateReshapedMark(mark, img, radius=128):
        """Generator for watermark mask

        Arguments:
            mark {ndarray} -- 1D PRND vector generated using pseudoGen method
            img {ndarray} -- host image

        Keyword Arguments:
            radius {int} -- radius at which to embed PRND vector (default: {128})

        Returns:
            {ndarray} -- watermark mask created with PRND vector embedded to given radius
        """
        watermark_mask = np.zeros(img.shape)
        length = len(mark)

        for ind in range(length):
            x1 = int((img.shape[0]/2) +
                     np.around(radius*math.cos(ind*math.pi/length)))
            y1 = int((img.shape[0]/2) +
                     np.around(radius*math.sin(ind*math.pi/length)))

            watermark_mask[x1, y1] = mark[ind]

        return WaterMark.extractMark(watermark_mask, radius)

    @staticmethod
    def frequency2Radius(img, frequencies):
        """convert frequency to radius

        Arguments:
            img {ndarray} -- image
            frequency {str} -- 'LOW', 'MEDIUM' or 'HIGH'

        Raises:
            AttributeError: Raised if wrong frequency zone is passed

        Returns:
            {float} -- radius representing provided frequency zone
        """
        # define radius for wanted zone
        if frequencies == 'LOW':
            radius = 1/8*(img.shape[0])
        elif frequencies == 'MEDIUM':
            radius = 1/4*(img.shape[0])
        elif frequencies == 'HIGH':
            radius = 3/8*(img.shape[0])
        else:
            raise AttributeError(
                "Unknown frequency zone. Use 'LOW', 'MEDIUM', or 'HIGH' ")

        return radius

    # TODO Make mark and vector real 1D-arrays, instead of 2D-arrays with empty 2nd dim.
    @staticmethod
    def covarMark(mark, vector):
        """Cross covariation of two 1D sequences. Measures the similarity between two vectors 
        by shifting (lagging) second vector as a function of the lag.

        Arguments:
            mark {ndarray} -- array like input sequences
            vector {ndarray} -- array like input sequences

        Returns:
            {ndarray} -- 1D vector of full cross covariation of two args
        """
        # normalize to zero mean
        mark = mark - np.mean(mark)
        vector = vector - np.mean(vector)

        vector_lenght = len(vector)
        max_cov = np.zeros(vector_lenght)
        counter = 0

        for i in range(len(vector)):
            vector_roll = np.roll(vector, counter)
            vector_2D = np.reshape(vector_roll, (vector_lenght, 1))
            max_cov[counter] = np.cov(mark[:, 0], vector_2D[:, 0])[0][1]
            counter += 1
        return np.amax(max_cov)

    @staticmethod
    def corrMark(mark, vector):
        """Cross correlation of two 1D sequences. Measures the similarity between two vectors 
        by shifting (lagging) second vector as a function of the lag.

        Arguments:
            mark {ndarray} -- array like input sequences
            vector {ndarray} -- array like input sequences

        Returns:
            {ndarray} -- 1D vector ofƒ full cross correlation of two args
        """
        # normalize to zero mean
        mark = mark - np.mean(mark)
        vector = vector - np.mean(vector)

        vector_lenght = len(vector)
        max_corr = np.zeros(vector_lenght)
        counter = 0

        for i in range(len(vector)):
            vector_roll = np.roll(vector, counter)
            vector_2D = np.reshape(vector_roll, (vector_lenght, 1))
            max_corr[counter] = np.corrcoef(mark[:, 0], vector_2D[:, 0])[0][1]
            counter += 1
        return np.amax(max_corr)

    @staticmethod
    def xcorrMark(mark, vector):
        """Plots the cross correlation of two 1D sequences that are first made zero-mean.

        Arguments:
            mark {ndarray} -- array like input sequences
            vector {ndarray} -- array like input sequences

        Returns:
            {ndarray} -- Plot of full cross correlation of two args
        """
        # normalize to zero mean
        mark = mark - np.mean(mark)
        vector = vector - np.mean(vector)
        correlation = plt.xcorr(mark[:, 0], vector[:, 0])
        return correlation

    @staticmethod
    def labQuality(imgZero, imgProcessed, profilePath, metric = 'PSNR'):
        """Method for transforming input CMYK images into LAB color space and computing their corresponding PSNR value.

        Arguments:
            imgZero {ndarray} -- image processed with zero Impact Factor value
            imgProcessed {ndarray} -- processed (marked or GCR) image
            profileName {str} -- profile name

        Returns:
            {float} -- PSNR value
        """
        # Creating cfAtoB object from gcrpywrap
        profpath = Path(profilePath)
        cfAtoB = gw.cform(profpath.resolve().as_posix(), 'AtoB1')

        # Reshape image arrays for cfAtoB method to work
        imgZeroReshape = np.reshape(imgZero, (512 * 512, 4))
        imgProcessedReshape = np.reshape(imgProcessed, (512*512, 4))

        # Transform image arrays for cfAtoB method to work
        imgZeroReshape = imgZeroReshape.astype(ctypes.c_double) / 2.55
        imgProcessedReshape = imgProcessedReshape.astype(
            ctypes.c_double) / 2.55

        # Create LAB from CMYK
        labOrig = cfAtoB.apply(imgZeroReshape)
        labMarked = cfAtoB.apply(imgProcessedReshape)

        # Normalize A and B channels to interval from 0 to 1
        labOrig[:, 1:3] = (labOrig[:, 1:3] + 128) / 255
        labMarked[:, 1:3] = (labMarked[:, 1:3] + 128) / 255

        # Normalize L channel to interval from 0 to 1
        labOrig[:, 0] = labOrig[:, 0] / 100
        labMarked[:, 0] = labMarked[:, 0] / 100

        # Delete object
        cfAtoB.delete()

        if metric == 'PSNR':
            # Return PSNR value
            return msr.peak_signal_noise_ratio(labOrig, labMarked, data_range=1)

        elif metric == 'SSIM':
            # Return SSIM value
            return msr.structural_similarity(labOrig, labMarked, multichannel=True)

        else:
            raise AttributeError('Unknown metric, please input PSNR or SSIM.')

    def findImpactFactor(self, img, rangePSNR=(38, 42), length=200, frequencies='MEDIUM', colorMode='CMYK'):
        """find impact factor that will return PSNR quality in given range

        Arguments:
            img {ndarray} -- host image 

        Keyword Arguments:
            rangePSNR {tuple of ints} -- range in dB (default: {(38 ,42)})
            length {int} -- length of embedded 1D vector (default: {200})
            frequencies {str} -- frequencies at which to search for vector ('LOW', 'MEDIUM', 'HIGH') (default: {'MEDIUM'})

        Returns:
            [float] -- impact factor for which encoded image will have given quality
        """
        # Defining range of Impact Factor values as a tuple
        arrImpFct = (50, 20000)

        # Defining left (lowest) and right (highest) values from tuple
        l = arrImpFct[0]
        r = arrImpFct[1]
        counter = 0
        # This while statement is just a measure of precaution. r <= l should be True
        while r >= l:
            counter += 1
            # print(counter)
            if counter >= 15:
                break
            # Defining the midpoint between l and r
            mid = l + (r - l) / 2

            # Setting Impact Factor value to midpoint of tuple
            impactFactor = mid

            # Calculating PSNR value
            imgMarked = self.embedMark(img, length, frequencies, impactFactor)
            # TODO Find a more elegant solution to solve input img and marked img PSNR
            imgBlind = self.embedMark(img, length, frequencies, 0)

            # TODO Check with Ante if this works
            if colorMode == 'CMYK':
                psnrMarked = msr.peak_signal_noise_ratio(imgBlind, imgMarked)
            elif colorMode == 'LAB':
                psnrMarked = self.labPSNR(imgBlind, imgMarked)
            else:
                raise AttributeError(
                    "Unknown color mode. Use 'CMYK', or 'LAB'.")

            if psnrMarked > rangePSNR[1]:
                l = mid
            elif psnrMarked < rangePSNR[0]:
                r = mid
            else:
                # print(psnrMarked)
                # print(impactFactor)
                return [impactFactor, psnrMarked]

        # If something unexpected occurs, return value of 1000
        # else:
        print("Error: Unable to find. Set Impact Factor to 1000.")
        print(psnrMarked)
        return [1000, psnrMarked]

    # TODO the method covert image to lab but accuracy is low.
    @staticmethod
    def profileCmyk2lab(img, profileCmyk):
        pilImg = Image.fromarray(img)
        labProfile = ImageCms.createProfile('LAB')
        cform = ImageCms.buildTransform(
            profileCmyk, labProfile, 'CMYK', 'LAB', renderingIntent=1)
        return np.array(ImageCms.applyTransform(pilImg, cform))

    @staticmethod
    def profileCmyk2srgb(img, profileCmyk):
        """convert from CMYK to sRGB color space using Cmyk profile

        Arguments:
            img {ndarray} -- image in CMYK color space
            profileCmyk {string} -- path to cmyk profile

        Returns:
            ndarray -- image in sRGB color space
        """
        pilImg = Image.fromarray(img)
        srgbProfile = ImageCms.createProfile('sRGB')
        cform = ImageCms.buildTransform(
            profileCmyk, srgbProfile, 'CMYK', 'RGB', renderingIntent=1)
        return np.array(ImageCms.applyTransform(pilImg, cform))

    @staticmethod
    def gcrMasking(orig, marked, profilePath):
        """GCR based method for masking artefacts introduced by watermark

        Arguments:
            orig {ndarray} -- original image in CMYK color space
            marked {ndarray} -- marked image in CMYK color space
            profileName {string} -- icc profile neaded for calculation 

        Returns:
            ndarray -- masked image
        """
        gcrmask = wmgcr.wmgcr(profilePath)
        # rpl = gcrmask.isReplaceable(orig,15,0)
        MethodGCR = 2
        imgMasked = gcrmask.transformImage(orig, marked, MethodGCR, 1, 0, 100)
        gcrmask.delete()
        return np.asarray(imgMasked)

    @staticmethod
    def parallelProcessing(func, srcFolder):
        """Utility method for concurent execution of a function on imgs in source folder

        Arguments:
            func {function} -- defined function that has only one parameter (img)
            srcFolder {string} -- path to folder with images

        Returns:
            dataFrame -- every row represents another image in folder, 
                         num of cols depend on defined function
        """
        # All TIF files in the src_path are now imgs
        listOfImgs = glob.glob(srcFolder+'*.tif')
        numImages = len(listOfImgs)

        my_q = PQueue()

        # no parameter provided in constructor, use all threads
        pool = multiprocessing.Pool()
        results = pool.map(func, listOfImgs)  # return data as list of arrays

        its = []
        while True:
            try:
                print("Waiting")
                i = my_q.get(True, 5)
                print("Found %s from the queue!" % i)
                its.append(i)
            except queue.Empty:
                print("Caught queue empty, done")
                break
        print("Processed %d items, completed." % len(its))

        pool.close()
        pool.join()

        # stack all rows fo the array to create dataframe
        return pd.DataFrame(np.row_stack(results))
