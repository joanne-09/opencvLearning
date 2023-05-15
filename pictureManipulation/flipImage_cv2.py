import cv2
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# flip image
# target: flip the image
# /*-------------*/

img = cv2.imread(f'{PATH}/sky1.jpg')
# change setting from BGR to RGB
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# flip the image
copy_img = RGB_img.copy()
img_flip = []
for flipcode in [0, 1, -1]:
    img_flip.append(cv2.flip(copy_img, flipcode))

# plot the three image
'''
plt.figure(figsize=(10, 6))

plt.subplot(131)
plt.imshow(img_flip[0])
plt.title('flipcode: 0')

plt.subplot(132)
plt.imshow(img_flip[1])
plt.title('flipcode: 1')

plt.subplot(133)
plt.imshow(img_flip[2])
plt.title('flipcode: -1')

plt.show()
'''

# rotate the image
copy_img_rotate = RGB_img.copy()
img_rotate = cv2.rotate(copy_img_rotate, 0)
# plt.imshow(img_rotate)
# plt.show()

copy_img_flip = RGB_img.copy()
# print(id(copy_img_flip), id(RGB_img))
flip = {"ROTATE_90_CLOCKWISE":cv2.ROTATE_90_CLOCKWISE,"ROTATE_90_COUNTERCLOCKWISE":cv2.ROTATE_90_COUNTERCLOCKWISE,"ROTATE_180":cv2.ROTATE_180}
for key, value in flip.items():
    plt.subplot(121)
    plt.imshow(copy_img_flip)
    plt.title('original')

    plt.subplot(122)
    plt.imshow(cv2.rotate(copy_img_flip, value))
    plt.title(key)
    plt.show()