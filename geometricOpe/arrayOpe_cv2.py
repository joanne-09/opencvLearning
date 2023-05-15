import matplotlib.pyplot as plt
import cv2
import numpy as np

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# array operation
# target: use array to change image with noise
# /*-------------*/

image_O = cv2.imread(f'{PATH}/desk.jpg')
# convert it to RGB
image = cv2.cvtColor(image_O, cv2.COLOR_BGR2RGB)
# 5830 3887 3
rows, cols, _ = image.shape

# change the pixel of the image
new_image = image + 20
# plot it!
new_image1 = 10*image
# plot it!

# create a array shape identical to image
# set the data type of Noise the same as image
Noise = np.random.normal(0, 20, (rows, cols, 3)).astype(np.uint8)

new_image2 = image + Noise
# plot it!

new_image3 = image * Noise
# plot it!
plt.imshow(new_image3)
plt.show()