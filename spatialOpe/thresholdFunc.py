import matplotlib.pyplot as plt
import cv2

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# thresholding
# target: threshold the image
# /*-------------*/

# load the image and convert to gray scale
image = cv2.imread(f'{PATH}/cat1.jpg',cv2.IMREAD_GRAYSCALE)

# src=the image to use
# thresh=the threshold
# maxval=the maxval to use
# type=type of filtering
ret, outs = cv2.threshold(src=image,thresh=0,maxval=255,
                          type=cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

plt.imshow(outs,cmap='gray')
plt.show()