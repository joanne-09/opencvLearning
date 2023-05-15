import cv2
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'
img = cv2.imread(f'{PATH}/sky2.jpg')
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# red image
img_red = RGB_img.copy()
# image[row, column, RGB layer]
img_red[:, :, 1] = 0
img_red[:, :, 2] = 0

# blue image
img_blue = RGB_img.copy()
img_blue[:, :, 0] = 0
img_blue[:, :, 1] = 0

# green image
img_green = RGB_img.copy()
img_green[:, :, 0] = 0
img_green[:, :, 2] = 0

# convert three images into one
img_colors = cv2.vconcat([img_red, img_green, img_blue])
# save the image
cv2.imwrite('picture/sky2_colors.jpg', img_colors)

plt.figure()
plt.subplot(121)
# draw the picture into the table
plt.imshow(RGB_img)
plt.subplot(122)
plt.imshow(img_colors)
# show it!
plt.show()