import cv2
import numpy as np
import helperFunc as helper

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# image sharpening
# target: use a kernel to sharpen the image
# /*-------------*/

# load the image
image = cv2.imread(f'{PATH}/cat1.jpg')
# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# resize it so that differences will be much noticeable
image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

# common kernel for image sharpening
# convolution kernel: 卷基層, 透過這個矩陣不斷在圖片上滑動再內積, 達到調整圖片的效果
kernel = np.array([[-1,-1,-1],
                   [-1,9,-1],
                   [-1,-1,-1]])
# filter2D func is to apply kernel(2D matrix) to the image
# src=image source
# ddepth=the size of the output image, -1 will remain the same size
# kernel=the matrix to apply for
sharpened = cv2.filter2D(src=image,ddepth=-1,kernel=kernel)
helper.plot_image(sharpened,image,title_1='Sharpeded image',title_2='Image')
# it seems to be sharpened
