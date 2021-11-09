from PIL import Image, ImageCms
import skimage.metrics as msr
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import math
from stego import stego_block

radius = 1/4 * 512
img_channel = 512

x1 = int((img_channel/2) + np.around(radius*math.cos(math.pi/1)))
y1 = int((img_channel/2) + np.around(radius*math.sin(math.pi/1)))

x2 = int((img_channel/2) + np.around(radius*math.cos(math.pi/1 + math.pi/8)))
y2 = int((img_channel/2) + np.around(radius*math.sin(math.pi/1 + math.pi/8)))

print('Done!')
# Saving image
# imgObject = Image.fromarray(img.astype('uint8'), 'CMYK')
# imgObject.save(imgName)
# print(f"Save merged image {img.shape}")

import cv2
img = cv2.imread('rsi_vjezba_3_slika.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.IMREAD_GRAYSCALE('rsi_vjezba_3_slika.jpg')
cv2.imwrite('rsi_vjezba_3_greyscale.jpg', img)

### ### ###
### ### ###
### ### ###

### ### ###
### ### ###
### ### ###

### ### ###
### ### ###
### ### ###

import classes

salary_employee = classes.SalaryEmployee(1, 'Ana Agić', 8000)
hourly_employee = classes.HourlyEmployee(2, 'Petar Jelušić', 6, 90)
commission_employee = classes.CommissionEmployee(3, 'Ante Poljičak', 14000, 1500)

payroll_system = classes.PayrollSystem()

payroll_system.calculate_payroll([
    salary_employee,
    hourly_employee,
    commission_employee
])

### ### ###
### ### ###
### ### ###

### ### ###
### ### ###
### ### ###

### ### ###
### ### ###
### ### ###

import stego_classes

encoder_1 = stego_classes.Encoder(
    pattern = 5, 
    message = 10, 
    permutation = 15, 
    length = 200, 
    frequency = 'MEDIUM', 
    strength = 5,
    block_number = 2,
    image_size = 512
)

encoder_2 = stego_classes.Encoder(
    pattern = 25, 
    message = 10, 
    permutation = 15, 
    length = 200, 
    frequency = 'MEDIUM', 
    strength = 5,
    block_number = 2,
    image_size = 512
)

decoder_1 = stego_classes.Decoder(
    pattern = 25, 
    message = 10, 
    permutation = 15, 
    length = 200, 
    frequency = 'MEDIUM', 
    strength = 5,
    block_number = 4,
    image_size = 512,
    channel = 'K'
)


encoder_1.show_properties()
encoder_2.show_properties()

decoder_1.show_properties()