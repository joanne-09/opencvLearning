from PIL import Image, ImageFilter
import numpy as np
import helperFunc as helper

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# linear filtering
# target: use a kernel to blur the image and reduce noise
# /*-------------*/

# load the image
image_O = Image.open(f'{PATH}/cat1.jpg')
rows_O, cols_O = image_O.size
image = image_O.resize((rows_O//4, cols_O//4))

# grab the size of image
# 1004 1504 x-axis y-axis
cols, rows = image.size
# create array (y-axis, x-axis)
# values are between 0~255
noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
# array can be add to image
# since image is stored as arrays
noisy_image = image + noise
# turn array into image
# since the attribute .filter only belongs to image
noisy_image = Image.fromarray(noisy_image)

# plot it first and see the differences
'''
helper.plot_image(image,noisy_image,title_1='Original',title_2='Image Plus Noise')
'''

# create a kernel which is a 5*5 array, each value is 1/36
# convolution kernel: 卷基層, 透過這個矩陣不斷滑動再內積, 達到調整圖片的效果
kernel = np.ones((5,5))/36
# ImageFilter.Kernel(size,kernel,scale=None,offset=0)
# size: kernel size
# sequence containing kernel weights
# scale: default is the sum of the kernel weight
# offset: the value is added to the result
# creates an convolution kernel
kernel_filter = ImageFilter.Kernel((5,5), kernel.flatten())
# filter performs a convolution between the image and the kernel on each color channel independently
image_filtered = noisy_image.filter(kernel_filter)

# plot it and see the outcome of the kernel
helper.plot_image(image_filtered,noisy_image,title_1='Filtered image',title_2='Image Plus Noise')
# it is expected to see that the noise is reduced but the image is blurry