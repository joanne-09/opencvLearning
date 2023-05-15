import matplotlib.pyplot as plt
import cv2

PATH = 'C:/Users/User/Documents/openCVLearning/picture'

image_O = cv2.imread(f'{PATH}/desk.jpg')
# convert it to RGB
image = cv2.cvtColor(image_O, cv2.COLOR_BGR2RGB)

# scale the horizontal axis by two and vertical axis as one
new_image = cv2.resize(image, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
# plot it!

# shrink the vertical axis as 0.5
new_image1 = cv2.resize(image, None, fx=1, fy=0.5, interpolation=cv2.INTER_CUBIC)
# plot it!

# specify the size of the resized image
rows = 100
cols = 200
new_image2 = cv2.resize(image, (100, 200), interpolation=cv2.INTER_CUBIC)

plt.imshow(new_image2)
plt.show()