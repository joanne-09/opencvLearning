import matplotlib.pyplot as plt
import cv2
import numpy as np

PATH = 'C:/Users/User/Documents/openCVLearning/picture'

image_O = cv2.imread(f'{PATH}/desk.jpg')
# convert it to RGB
image = cv2.cvtColor(image_O, cv2.COLOR_BGR2RGB)

# translation
# shift the location of the image
# tx is the number of pixels you shift the location
# ty is the number of pixels you shift in the vertical direction
tx = 100
ty = 0
M = np.float32([[1,0,tx], [0,1,ty]])

# the shape of the image
rows, cols, _ = image.shape
# in order not to cut the image
# we will add tx(ty) to cols(rows)
# warpAffine(<image array>, transformation matrix <M>, the length and width of the output image<(col, row)>)
new_image = cv2.warpAffine(image, M, (cols+tx, rows))
# plot it!
plt.imshow(new_image)
plt.show()