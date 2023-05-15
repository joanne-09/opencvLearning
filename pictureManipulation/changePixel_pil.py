from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# change certain pixel
# target: directly change pixels
# /*-------------*/

# because we use PIL to open instead of cv2
# the setting is RGB
img = Image.open(f'{PATH}/desk.jpg')
# turn image into array
arr = np.array(img)

# set number of pixel
upper = 400
lower = 1000
left = 400
right = 1000

# copy of image
arr_copy = np.copy(arr)
# set the assigned part to only R&B layer
# array[row, column, RGB layer]
arr_copy[upper:lower, left:right, 1:2] = 0

'''
plt.figure()
plt.subplot(121)
plt.imshow(arr)
plt.title('Original')
plt.subplot(122)
plt.imshow(arr_copy)
plt.title('Altlered Image')
plt.show()
'''

# copy an image
img_draw = img.copy()
# whatever method we apply to the object img_fn
# will change the image object img_draw
# TODO: I still don't know why it works
img_fn = ImageDraw.Draw(im=img_draw)
# set a shape
shape = [left, upper, right, lower]
# draw a rectangle on the specified location
img_fn.rectangle(xy=shape, fill='red')

# method one
# another image
PATH = 'C:/Users/User/Documents/openCVLearning/picture'
img_sky = Image.open(f'{PATH}/sky1.jpg')
arr_sky = np.array(img_sky)
# just set the array to another
arr_sky[upper:lower,left:right,:]=arr[upper:lower,left:right,:]

# method two
# crop the image
crop_img = img.crop((left, upper, right, lower))
# paste previous image to the new one
img_sky.paste(crop_img, box=(left, upper))

plt.figure()
plt.imshow(img_sky)
plt.show()