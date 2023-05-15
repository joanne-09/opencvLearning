import cv2
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'
img = cv2.imread(f'{PATH}/sky2.jpg')
# set the size of the table default(6, 4)
plt.figure()
# draw the picture in the table
plt.imshow(img)
# show it!
plt.show()