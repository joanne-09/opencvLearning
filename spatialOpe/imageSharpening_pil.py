from PIL import Image, ImageFilter
import numpy as np
import helperFunc as helper

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# sharpen image
# target: use a kernel to sharpen the image
# /*-------------*/

# load the image
image_O = Image.open(f'{PATH}/cat1.jpg')
rows_O, cols_O = image_O.size
# resize the image so that differences will be more noticeable
image = image_O.resize((rows_O//4, cols_O//4))

# common kernel for image sharpening
# convolution kernel: 卷基層, 透過這個矩陣不斷在圖片上滑動再內積, 達到調整圖片的效果
kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])
# ImageFilter.Kernel(size,kernel,scale=None,offset=0)
# size: kernel size
# sequence containing kernel weights
# scale: default is the sum of the kernel weight
# offset: the value is added to the resul
# creates an convolution kernel
kernel = ImageFilter.Kernel((3,3), kernel.flatten())
# apply the filter on the image
sharpened = image.filter(kernel)
# plot it!
helper.plot_image(sharpened,image,title_1='Sharpened image',title_2='Image')
# it turns out that the image sharpens

# using build-in method to sharpen the image
sharpened1 = image.filter(ImageFilter.SHARPEN)
# plot it!
helper.plot_image(sharpened1,image,title_1='Sharpened image',title_2='Image')
# it turns out that the image sharpens
# but not that effective as previous method
