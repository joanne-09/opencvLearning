import matplotlib.pyplot as plt
import cv2

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# edge filter
# target: grab the edge of picture and plot it
# /*-------------*/

# load the image and turn it into gray scale
img_gray = cv2.imread(f'{PATH}/cat2.jpg', cv2.IMREAD_GRAYSCALE)
# resize it so that differences will be much noticeable
img_gray = cv2.resize(img_gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

ddepth = cv2.CV_16S
# apply the filter on the image in the X direction
# get vertical edge
# Sobel(<src> input image, <ddepth> output image depth,
#       <dx> order of derivative x, <dy> order of derivative y,
#       <ksize> size of the extended Sobel kernel must be 1,3,5, or 7)
grad_x = cv2.Sobel(src=img_gray,ddepth=ddepth,dx=1,dy=0,ksize=3)
# apply the filter on the image in the Y direction
# get horizontal edge
grad_y = cv2.Sobel(src=img_gray,ddepth=ddepth,dx=0,dy=1,ksize=3)

# convert the image back to uint8(0 n 255)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
# combine images of two direction
grad = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
# these two methods are used with Sobel

plt.imshow(grad,cmap='gray')
plt.title('Edged image')
plt.show()


# /*-------------*/
# median filter
# target: use medianBluc func to filter noise
# /*-------------*/

# load the image and convert to gray scale
image = cv2.imread(f'{PATH}/cat1.jpg', cv2.IMREAD_GRAYSCALE)
# resize it
image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

# filter the image using a kernel of size 5
filtered_image = cv2.medianBlur(image, 5)
# blur the image with kernel

plt.imshow(filtered_image,cmap='gray')
plt.title('Median blurred image')
plt.show()