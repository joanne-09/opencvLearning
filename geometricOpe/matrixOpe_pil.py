from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

# type ctrl+F 'CONCLUSION: ' to easily find my learnings
PATH = 'C:/Users/User/Documents/openCVLearning/picture'


# /*-------------*/
# matrix operation
# target: use matrix to modify image
# /*-------------*/

# RGB color
desk_img = Image.open(f'{PATH}/desk.jpg')
# turn into array
arr_desk_img = np.array(desk_img)
# crop it to meet the need
# use three parameter to crop it [row, col, layer]
crop_desk_img = arr_desk_img[:3000, :3000,:]

sky_img = Image.open(f'{PATH}/sky1.jpg')
# convert the image into gray scale
sky_gray = ImageOps.grayscale(sky_img)
width, height = sky_gray.size
# turn into array
arr_sky_gray = np.array(sky_gray)
# crop it to meet the need
# since it only has one layer
# you only need two parameter
crop_arr_sky_gray = arr_sky_gray[:3000, :3000]

# linalg.svd(a<2D array>, full_matrices=False)
# linalg.svd(a<other array>, full_matrices=True)
# a is a (M, N) array, a=u*s*v
# return u:unitary arrays. (M, M)
#        s:vectors. (M, N)
#        v:unitary arrays. (N, N)
U, s, V = np.linalg.svd(crop_arr_sky_gray, full_matrices=True)
# (3887,)
# print(s.shape)

# TODO: either, i don't know how it works
S = np.zeros((crop_arr_sky_gray.shape[0], crop_arr_sky_gray.shape[1]))
S[:crop_desk_img.shape[0], :crop_desk_img.shape[0]] = np.diag(s)
# helper.plot_image(U, V, title1='Matrix U', title2='Matrix V')

# we can plot an image array
# but it is still an array
# therefore there are no image attributes
'''
plt.imshow(S, cmap='gray')
plt.show()
'''

# perform matrix multiplication on S and V
# assign it to B
B = S.dot(V)
# do the same to U & B and assign it to A
A = U.dot(B)
# which shows the whole image(of course, cropped)
'''
plt.imshow(A, cmap='gray')
plt.show()
'''
# which means the original image=U*S*V

# change the resolution of the image
# by split it and merge
# TODO: either, i don't know how it works
for n_component in [1,10,100,200,500]:
    S_new=S[:,:n_component]
    V_new=V[:n_component,:]
    A_new=U.dot(S_new.dot(V_new))
    plt.imshow(A_new, cmap='gray')
    plt.title('Number of Components:'+str(n_component))
    plt.show()