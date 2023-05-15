import cv2
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'
img = cv2.imread(f'{PATH}/sky2.jpg')

new_img = img.copy()

plt.figure()
rows = 256
plt.subplot(121)
# draw the image into the table
# image[row, colums, RGB layer]
plt.imshow(new_img[0:rows,:,:])
columns = 256
plt.subplot(122)
plt.imshow(new_img[:,0:columns,:])
# show it
plt.show()