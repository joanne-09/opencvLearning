from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# crop image
# target: crop the image
# /*-------------*/

# because we use PIL to open instead of cv2
# the setting is RGB
img = Image.open(f'{PATH}/desk.jpg')
arr = np.array(img)

upper = 400
lower = 1000
# [row, column, RGB layer]
crop_top = arr[upper: lower, :, :]

left = 400
right = 1000
crop_horizontal = crop_top[:, left:right, :]

# build in method
crop_img = img.crop((left, upper, right, lower))

plt.figure()
plt.imshow(crop_img)
plt.show()