from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# flip image
# target: flip the image
# /*-------------*/

# because we use PIL to open instead of cv2
# the setting is RGB
img = Image.open(f'{PATH}/sky1.jpg')

arr = np.array(img)
width, height, C = arr.shape
# 5830 3887 3
# print(width, height, C)

# traditional flip method
arr_flip = np.zeros((width, height, C), dtype=np.uint8)
for i, row in enumerate(arr):
    arr_flip[width-1-i, :, :] = row

# use ImageOps method to flip and mirror
img_flip = ImageOps.flip(img)
img_mirror = ImageOps.mirror(img)
# print(id(img), id(img_flip))

# another method to flip upside down
# the Image module has built-in attributes that describe the typr of flip
img_flip1 = img.transpose(1)

plt.figure()
plt.imshow(img_flip1)
plt.show()