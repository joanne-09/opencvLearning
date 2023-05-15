import matplotlib.pyplot as plt
import cv2
import numpy as np
import helperFunc as helper

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# matrix operation
# target: use matrix to modify image
# /*-------------*/

image_O = cv2.imread(f'{PATH}/desk.jpg')
# convert it to RGB
image_RGB = cv2.cvtColor(image_O, cv2.COLOR_BGR2RGB)
image = image_RGB[:3000, :3000, :]
# 5830 3887 3
rows, cols, _ = image.shape

img_gray_O = cv2.imread(f'{PATH}/sky1.jpg', cv2.IMREAD_GRAYSCALE)
img_gray = img_gray_O[:3000, :3000]

# linalg.svd(a<2D array>, full_matrices=False)
# linalg.svd(a<other array>, full_matrices=True)
# a is a (M, N) array, a=u*s*v
# return u:unitary arrays. (M, M)
#        s:vectors. (M, N)
#        v:unitary arrays. (N, N)
U, s, V = np.linalg.svd(img_gray, full_matrices=True)
# plot it!
helper.plot_image(U,V,'Matrix U','Matrix V')

# TODO: either, i don't know how it works
S = np.zeros((img_gray.shape[0], img_gray.shape[1]))
S[:image.shape[0], :image.shape[0]] = np.diag(s)
plt.imshow(S, cmap='gray')
plt.show()

# perform matrix multiplication on S and V
# assign it to B
B = S.dot(V)
A = U.dot(B)
# which shows the whole image(of course, cropped)
plt.imshow(A, cmap='gray')
plt.show()
# which means the original image=U*S*V

# change the resolution of the image
# by split it and merge
for n_component in [1,10,100,200,500]:
    S_new=S[:,:n_component]
    V_new=V[:n_component,:]
    A_new=U.dot(S_new.dot(V_new))
    plt.imshow(A_new, cmap='gray')
    plt.title('Number of Components:'+str(n_component))
    plt.show()