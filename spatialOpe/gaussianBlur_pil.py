from PIL import Image, ImageFilter
import numpy as np
import helperFunc as helper

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# gaussian blur
# target: use func to filter noise by blurring the image
# /*-------------*/

# load the image
image_O = Image.open(f'{PATH}/cat1.jpg')
rows_O, cols_O = image_O.size
# resize image so that differences are more noticeable
image = image_O.resize((rows_O//4, cols_O//4))

# grab the size of image
rows, cols = image.size
print(rows, cols)
# create array (x-axis, y-axis)
noise = np.random.normal(0,15,(cols,rows,3)).astype(np.uint8)
# CONCLUSION: you can add array to an image
# CONCLUSION: and the output will become an array
noisy_image = image + noise
# test Image; array; array
'''
print(type(image))
print(type(noise))
print(type(noisy_image))
'''
# turn array into image
# in order to use the attributes of image
noisy_image = Image.fromarray(noisy_image)

# plot it first and see the differences
'''
helper.plot_image(image,noisy_image,title_1='Original',title_2='Image Plus Noise')
'''

# filter the image using builn-in method
# only image has this method
# image.filter(ImageFilter.GaussianBlur(<int> blur kernel radius, default 2))
image_filtered = noisy_image.filter(ImageFilter.GaussianBlur)
# try using para=4, the image is more blurred
image_filtered1 = noisy_image.filter(ImageFilter.GaussianBlur(4))

helper.plot_image(image_filtered1,noisy_image,title_1='Filtered image',title_2='Image Plus Noise')
# it turns out that the noise is reduced but the image is blurred