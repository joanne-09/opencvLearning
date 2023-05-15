import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_image(img1, img2, title1='Original', title2='New Image'):
    plt.figure()

    plt.subplot(121)
    plt.imshow(img1,cmap='gray')
    plt.title(title1)

    plt.subplot(122)
    plt.imshow(img2,cmap='gray')
    plt.title(title2)

    plt.show()


def plot_hist(old_img,new_img,title_old='Original',title_new='New Image'):
    intensity_values=np.array([x for x in range(256)])
    plt.figure()

    plt.subplot(121)
    # use this func to generate histogram
    # x-axis:0~255 y-axis:intensity of each value
    # cv2.calcHist(CV array:[image], channel:[0], always be [None],
    #              the number of bins:[L], the range of index of bins:[0,L-1])
    plt.bar(intensity_values,cv2.calcHist([old_img],[0],None,[256],[0,256])[:,0],width=5)
    plt.title(title_old)
    plt.xlabel('intensity')

    plt.subplot(122)
    plt.bar(intensity_values,cv2.calcHist([new_img],[0],None,[256],[0,256])[:,0],width=5)
    plt.title(title_new)
    plt.xlabel('intensity')

    plt.show()


def thresholding(input_img,threshold,max_value=255,min_value=0):
    N, M = input_img.shape
    image_out=np.zeros((N,M),dtype=np.uint8)

    for i in range(N):
        for j in range(M):
            if input_img[i, j]>threshold:
                image_out[i,j]=max_value
            else:
                image_out[i,j]=min_value
    
    return image_out