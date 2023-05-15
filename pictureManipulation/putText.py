import cv2
import matplotlib.pyplot as plt

PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# put text
# target: put certain text into an image
# /*-------------*/

img = cv2.imread(f'{PATH}/sky2.jpg')
# change setting from BGR to RGB
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# image shape 2568 4256

# put text into my image
# putText(imageName, text, position, color(RGB), font, fontSize, thicknessOfText)
cv2.putText(img=RGB_img,text='Sky',org=(1800, 1400),color=(0,0,0),fontFace=4,fontScale=20,thickness=5)

plt.figure()
plt.imshow(RGB_img)
plt.show()