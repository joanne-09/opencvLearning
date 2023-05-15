import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# change certain pixel
# target: change pixels directly into icons
# /*-------------*/

img = cv2.imread(f'{PATH}/sky1.jpg')
# change setting from BGR to RGB
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

upper = 400
lower = 1000
left = 400
right = 1000

# change the pixel into a black rectangle
arr = np.copy(RGB_img)
arr[upper:lower, left:right,:]=0

# for previous practice
'''
plt.figure()
plt.subplot(121)
plt.imshow(RGB_img)
plt.title('original')

plt.subplot(122)
plt.imshow(arr)
plt.title('Altered Image')

plt.show()
'''

start_point, end_point = (left, upper), (right, lower)
img_draw = np.copy(RGB_img)
# draw a rectangle
cv2.rectangle(img_draw,pt1=start_point,pt2=end_point,color=(0,255,0),thickness=5)
plt.figure()
plt.imshow(img_draw)
plt.show()