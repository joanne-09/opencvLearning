import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# edge filter
# target: use filter to find the edge of picture
# /*-------------*/

img_gray = Image.open(f'{PATH}/cat2.jpg')
# convert image to gray scale
img_gray = img_gray.convert('L')
rows_O, cols_O = img_gray.size
img_gray = img_gray.resize((rows_O//4, cols_O//4))

# they can only apply on grayscale images
# filter the image
img_gray1 = img_gray.filter(ImageFilter.EDGE_ENHANCE)
img_gray2 = img_gray.filter(ImageFilter.FIND_EDGES)

plt.imshow(img_gray1, cmap='gray')
plt.show()


# /*-------------*/
# median filter
# target: use MedianFilter to filter the noise and blur the background
# /*-------------*/

image_O = Image.open(f'{PATH}/cat1.jpg')
# convert to gray scale
image_O = image_O.convert('L')
rows_O, cols_O = image_O.size
image = image_O.resize((rows_O//4, cols_O//4))

# blurs the background and increase the segmentation
image = image.filter(ImageFilter.MedianFilter)
plt.imshow(image,cmap='gray')
plt.show()