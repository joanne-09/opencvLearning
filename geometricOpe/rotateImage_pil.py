from PIL import Image
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'

desk_img = Image.open(f'{PATH}/desk.jpg')

# build in func to rotate image
# rotate counterclockwise
theta = 45
new_desk_img = desk_img.rotate(theta)
plt.imshow(new_desk_img)
plt.show()