import cv2
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# crop image
# target: crop the image
# /*-------------*/

img = cv2.imread(f'{PATH}/sky1.jpg')
# change setting from BGR to RGB
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

upper = 400
lower = 1000
left = 400
right = 1000

# crop the image
# image[row, col, RGB layer]
crop_top = RGB_img[upper:lower,:,:]
crop_horizontal = crop_top[:,left:right,:]
plt.figure()
plt.imshow(crop_horizontal)
plt.show()