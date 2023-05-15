import cv2
import helperFunc as helper

PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# adjust brightness and contrast
# target: adjust brightness(beta) and contrast(alpha)
# /*-------------*/

# set image as RGB
desk_img = cv2.imread(f'{PATH}/desk.jpg')
# convert BGR to RGB image
desk_RGB = cv2.cvtColor(desk_img, cv2.COLOR_BGR2RGB)

# only adjust brightness
# simple contrast control
alpha = 1
# simple brightness control(brighter)
beta = 100
# set a new image and set different brightness
# all pixel value are between 0 n 255(3 byte)
new_desk_RGB_bright = cv2.convertScaleAbs(desk_RGB,alpha=alpha,beta=beta)
# plot image
helper.plot_image(desk_RGB, new_desk_RGB_bright, title1='Original',title2='brightness control')
# turn into histogram to see the difference
helper.plot_hist(desk_RGB, new_desk_RGB_bright,'Original','brightness control')

# only adjust contrast
# simple contrast control(sharper)
alpha=2
# simple brightness control
beta=0
# set a new image and set different sharpness
# all pixel value are between 0 n 255(3 byte)
new_desk_RGB_sharp = cv2.convertScaleAbs(desk_RGB,alpha=alpha,beta=beta)
# plot image
helper.plot_image(desk_RGB, new_desk_RGB_sharp, title1='Original',title2='contrast control')
# turn into histogram to see the difference
helper.plot_hist(desk_RGB, new_desk_RGB_sharp,'Original','contrast control')

# adjust contrast and brightness
alpha=3
beta=-200
new_img = cv2.convertScaleAbs(desk_RGB, alpha=alpha, beta=beta)
helper.plot_image(desk_RGB, new_img, 'Original', 'brightness & contrast control')
helper.plot_hist(desk_RGB, new_img, 'Original', 'brightness & contrast control')

# increase the contrast of image
# by stretching out the range of the grayscale pixels
new_desk_equalize = cv2.equalizeHist(desk_RGB)
helper.plot_image(desk_RGB, new_desk_equalize, 'Original', 'Histogram Equalization')
helper.plot_hist(desk_RGB, new_desk_equalize, 'Original', 'Histogram Equalization')