import cv2
import numpy as np
import helperFunc as helper

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# gaussian blur
# target: use GaussianBlur to filter the noise
# /*-------------*/

image = cv2.imread(f'{PATH}/cat1.jpg')
# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# resize it so that differences will be much noticeable
image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

# y-axix, x-axis
rows, cols, _ = image.shape
# create array (y-axis, x-axis)
# values are between 0~255 since we only use 3 byte
noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
# image plus array
# and return an array
noisy_image = image + noise
# plot the array as image
helper.plot_image(image,noisy_image,title_1='Original',title_2='Image Plus Noise')

# this method will smoothen the edge of picture
# src=input image
# ksize=kernel size
# sigmaX=standard deviation in the X direction
# sigmaY=standard devaition in the Y direction
image_filtered=cv2.GaussianBlur(noisy_image,(5,5),sigmaX=4,sigmaY=4)
helper.plot_image(image_filtered,noisy_image,title_1='Filtered image(5x5)',title_2='Image Plus Noise')
# blurred and noise is reduced

image_filtered1=cv2.GaussianBlur(noisy_image,(11,11),sigmaX=4,sigmaY=4)
helper.plot_image(image_filtered1,noisy_image,title_1='Filtered image(11x11)',title_2='Image Plus Noise')
# more blurred
