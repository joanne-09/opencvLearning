from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# array operation
# target: use array to change the image
# /*-------------*/

desk_img = Image.open(f'{PATH}/desk.jpg')
width, height = desk_img.size

# turn image into array
arr_desk_img = np.array(desk_img)

# change pixel of image
# CONCLUSION: in this case, we need array to add an integer to pixels
# CONCLUSION: since you can never add a number to image
new_arr_desk_img = arr_desk_img + 20
new_arr_desk_img1 = 10 * arr_desk_img

# add elements of two arrays of equal shape
# an array of random noises with the same shape and data type as our image
Noise = np.random.normal(0, 20, (height, width, 3)).astype(np.uint8)

# CONCLUSION: but an image can add an array
# CONCLUSION: probable is that an image is stored as arrays
# CONCLUSION: therefore the computer can understand it

# randomly add integer to each pixel
new_arr_desk_img_with_noise = arr_desk_img + Noise
# randomly multyply integer to each pixel
new_arr_desk_img_with_noise1 = arr_desk_img * Noise

plt.imshow(new_arr_desk_img_with_noise1)
plt.show()