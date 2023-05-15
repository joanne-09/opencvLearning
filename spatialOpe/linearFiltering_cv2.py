import cv2
import numpy as np
import helperFunc as helper

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# linear filtering
# target: use a kernel to blur the image and reduce noise
# /*-------------*/

# load an image
image = cv2.imread(f'{PATH}/cat1.jpg')
# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# resize it so that changes are more apparent
image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

# y-axix, x-axis, layers
rows, cols, _ = image.shape
# create array (y-axis, x-axis)
# values are between 0~255 since we only use three byte to store int
noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
# CONCLUSION: image can add to array, since image is stored as array in computer
# CONCLUSION: then return an array
noisy_image = image + noise
# CONCLUSION: an array can be plot as image
helper.plot_image(image,noisy_image,title_1='Original',title_2='Image Plus Noise')

# create a kernel
# convolution kernel: 卷基層, 透過這個矩陣不斷滑動再內積, 達到調整圖片的效果
kernel = np.ones((6,6))/36
# filter the image using the kernel
# src=image source
# ddepth=the size of the output image, -1 will remain the same size
image_filtered = cv2.filter2D(src=noisy_image,ddepth=-1,kernel=kernel)
# plot it!
helper.plot_image(image_filtered,noisy_image,title_1='Filtered image(6x6)',title_2='Image Plus Noise')
# it turns out that the noise is reduced but the image is blurred

# create a small kernel
kernel1 = np.ones((4,4))/16
# filter the image using the kernel
image_filtered1 = cv2.filter2D(src=noisy_image,ddepth=-1,kernel=kernel1)
# plot it!
helper.plot_image(image_filtered1,noisy_image,title_1='Filtered image(4x4)',title_2='Image Plus Noise')
# it turns out that the noise is reduced but the image is blurred
# a smaller kernel keeps the image sharp but filters less noise
