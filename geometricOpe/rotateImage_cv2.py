import matplotlib.pyplot as plt
import cv2
import numpy as np
import helperFunc as helper

PATH = 'C:/Users/User/Documents/openCVLearning/picture'

# 0:blcak 255:white
toy_image = np.zeros((6,6))
toy_image[1:5,1:5]=255
toy_image[2:4,2:4]=0
# plot it
# cmap='gray'

# rotate toy image by 45 degree
theta = 45.0
# getRotationMatrix2D(center of the rotation in the source image <(col,row)>,
#                     rotation angle degree <int>,
#                     isotropic scale factor <(col,row)>)
M = cv2.getRotationMatrix2D(center=(3,3), angle=theta, scale=1)
# warpAffine(<image array>, transformation matrix <M>, the length and width of the output image<(col, row)>)
new_toy_image = cv2.warpAffine(toy_image, M, (6,6))
helper.plot_image(toy_image, new_toy_image, title1='Original', title2='rotated image')

image_O = cv2.imread(f'{PATH}/desk.jpg')
# convert it to RGB
image = cv2.cvtColor(image_O, cv2.COLOR_BGR2RGB)

cols, rows, _=image.shape

# rotate the image by 45 degree countercolckwise
M_new = cv2.getRotationMatrix2D(center=(cols//2 -1,rows//2 -1), angle=theta, scale=1)
new_image = cv2.warpAffine(image, M_new, (cols, rows))
plt.imshow(new_image)
plt.show()