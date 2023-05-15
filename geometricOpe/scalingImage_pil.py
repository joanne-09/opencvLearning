from PIL import Image
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'

desk_img = Image.open(f'{PATH}/desk.jpg')

# grab the size of the image
width, height = desk_img.size

# scaling
# double the width of the image
new_width = 2*width
new_height = height
new_desk_img = desk_img.resize((new_width, new_height))

# double the height of the image
new_width1 = width
new_height1 = 2*height
new_desk_img1 = desk_img.resize((new_width1, new_height1))

# shrink the image's width and height both by 1/2
new_width2 = width//2
new_height2 = height//2
new_desk_img2 = desk_img.resize((new_width2, new_height2))
plt.imshow(new_desk_img2)
plt.show()