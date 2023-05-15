import cv2
import matplotlib.pyplot as plt
import numpy as np

PATH = 'C:/Users/User/Documents/openCVLearning/picture'

# /*--------------------*/
# toy_image
# a normal one and a negative one
# /*--------------------*/

# create a histogram, a grid image
# 0:black 1:gray 2:white
toy_image=np.array([[0,2,2],
                    [1,1,1],
                    [1,1,2]],dtype=np.uint8)
'''
plt.imshow(toy_image, cmap='gray')
plt.show()
print('toy_image:', toy_image)
'''

# turn into a negative one
neg_toy_image = -1*toy_image +255
'''
print('toy image\n', neg_toy_image)
print('image negatives\n', neg_toy_image)
'''

# white turns to black
# black turns to white
'''
plt.figure()
plt.subplot(121)
plt.imshow(toy_image,cmap='gray')
plt.subplot(122)
plt.imshow(neg_toy_image,cmap='gray')
plt.show()
print('toy_image:', toy_image)
'''


# /*--------------------*/
# import a sky image and turn into gray scale
# draw a bar graph and a line graph
# show intensity of every pixel value
# /*--------------------*/

# draw a bar graph
# plt.bar(x-axis, y-axis)
'''
plt.bar([x for x in range(6)],[1,5,2,0,0,0])
plt.show()
'''

# import an image, turn into gray scale
sky=cv2.imread(f'{PATH}/sky1.jpg',cv2.IMREAD_GRAYSCALE)

# represent the distribution of pixel intensities
# pixel values will be in the range of 0~255
# x-axis serves as 'bins', will be distribute into 255 bins, count the number of times each pixel value orrurs
# cv2.calcHist(CV array:[image], channel:[0], always be [None],
#              the number of bins:[L], the range of index of bins:[0,L-1])
hist=cv2.calcHist([sky],[0],None,[256],[0,256])
# hist=[[intensity1], [intensiry2], ....]

# bar graph
# x-axis are the pixel intensity (0~255)
# y-axis is the number of times of occurrences that the corresponding pixel intensity value
intensity_values = np.array([x for x in range(hist.shape[0])])
# plt.bar(x-axis, y-axis, width)
# hist[:,0] = those intensities
'''
plt.bar(intensity_values, hist[:,0], width=5)
plt.title('Bar histogram')
plt.show()
'''

# convert bar to line graph
# normalize it by the number of pixels
PMF = hist/(sky.shape[0]*sky.shape[1])
'''
plt.plot(intensity_values, hist)
plt.title('histogram')
plt.show()
'''

# draw a rectangle in the gray scale sky image
sky_new = sky.copy()
cv2.rectangle(sky_new,pt1=(160,1000),pt2=(250,800),color=(255),thickness=5)
# draw a negative image
neg_sky_new=-1*sky_new + 255
'''
plt.figure()
plt.imshow(neg_sky_new, cmap='gray')
plt.show()
'''


# /*--------------------*/
# import RGB desk image
# split RGB channel and draw pixel intensity of each layer
# /*--------------------*/

# import a image and convert it to RGB
desk_img = cv2.imread(f'{PATH}/desk.jpg')
desk_RGB = cv2.cvtColor(desk_img, cv2.COLOR_BGR2RGB)

# line graph
color = ('red','green','blue')
# i:what color channel is
# col:the value of color
for i,col in enumerate(color):
    # i: switch color channel
    histr = cv2.calcHist([desk_RGB],[i],None,[256],[0,256])
    # plt.plot(x-axis,y-axis,line color, label name)
    plt.plot(intensity_values,histr,color=col,label=col+' channel')
    # set or get x-axis limit
    plt.xlim([0,256])

# place a legend(label) in the table
plt.legend()
plt.title('Histogram Channels')
plt.show()