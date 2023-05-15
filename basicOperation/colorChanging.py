import cv2
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'
img = cv2.imread(f'{PATH}/sky2.jpg')
# assign a new image and change the color setting
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# change the color setting to gray
# however it still looks colorful
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print the size of the image
print(img_gray.shape)
# if the image would like to be gray scale
# cmap='gray'
# cmap -> colormap, set this variable and will output specific color dict
# finally set the image to gray scale
plt.imshow(img_gray, cmap='gray')
# save the image
cv2.imwrite('picture/sky2_gray.jpg', img_gray)
plt.show()