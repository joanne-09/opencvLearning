import cv2
import numpy as np
import matplotlib.pyplot as plt
import helperFunc as helper

PATH = 'C:/Users/User/Documents/openCVLearning/picture'
sky_img = cv2.imread(f'{PATH}/sky1.jpg', cv2.IMREAD_GRAYSCALE)


# /*-------------*/
# image thresholding
# target:set a threshold and transform the image
# /*-------------*/

# dtype -> data type
# be used as functions to convert python numbers to array scalars
# uint8 means the storage of int is 8 byte(0~255)(total 2^8 intager)
toy_image=np.array([[0,2,2],
                    [1,1,1],
                    [1,1,2]],dtype=np.uint8)
threshold = 1
max_value = 2
min_value = 0
thresholding_toy=helper.thresholding(toy_image, threshold=threshold, max_value=max_value, min_value=min_value)
'''
plt.figure()
plt.subplot(121)
plt.imshow(toy_image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(thresholding_toy, cmap='gray')
plt.title('Image after Thresholdng')

plt.show()
'''


gray_sky_copy=sky_img.copy()

intensity_values=[x for x in range(256)]
# draw a histogram
# cv2.calcHist([pic name], [color channel], mask usually [None], [bins], [bin label])
hist=cv2.calcHist([gray_sky_copy],[0],None,[256],[0, 255])
# plt.bar(x-axis, y-axis)
'''
plt.bar(intensity_values, hist[:, 0], width=5)
plt.title('Bar histogram')
plt.show()
'''


# threshold the sky image
# this is convertable
threshold1 = 87
threshold_gray_sky_copy = helper.thresholding(gray_sky_copy, threshold=threshold1, max_value=255, min_value=0)
helper.plot_image(gray_sky_copy, threshold_gray_sky_copy, 'Original', 'Image After Thresholding')
helper.plot_hist(gray_sky_copy, threshold_gray_sky_copy, 'Original', 'Image After Thresholding')


# cv2 build in threshold func
ret, cv2_threshold_gray_sky_copy=cv2.threshold(gray_sky_copy, threshold1, 255, cv2.THRESH_BINARY)
helper.plot_image(gray_sky_copy, cv2_threshold_gray_sky_copy, 'ORiginal', 'Image After Thresholding')
helper.plot_hist(gray_sky_copy, cv2_threshold_gray_sky_copy, 'ORiginal', 'Image After Thresholding')


# another cv2 build in func
ret, otsu_threshold_gray_sky_copy=cv2.threshold(gray_sky_copy,0,255,cv2.THRESH_OTSU)
helper.plot_image(gray_sky_copy, otsu_threshold_gray_sky_copy,'Original','Otsu')
helper.plot_hist(gray_sky_copy, otsu_threshold_gray_sky_copy,'Original','Otsu\' method')