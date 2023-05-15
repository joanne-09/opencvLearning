import cv2
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'
img = cv2.imread(f'{PATH}/sky2.jpg')
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# split the image into three channel
# image[row, column, RGB layer]
blue, green, red = RGB_img[:, :, 0], RGB_img[:, :, 1], RGB_img[:, :, 2]
# use vconcat to convert three image into one
BGR_img_split = cv2.vconcat([blue, green, red])

# the figure of the table
# assign a new table
plt.figure()
# subplot(row, colomn, index)
# images' row and colomn must be same the the same table
plt.subplot(121)
plt.imshow(RGB_img)
# set the title of the left image
plt.title('RGB image')
# plot another image the the same table
plt.subplot(122)
plt.imshow(BGR_img_split ,cmap='gray')
plt.show()