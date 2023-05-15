import matplotlib.pyplot as plt

def plot_image(img1, img2, title1='Original', title2='New Image'):
    plt.figure()

    plt.subplot(121)
    plt.imshow(img1,cmap='gray')
    plt.title(title1)

    plt.subplot(122)
    plt.imshow(img2,cmap='gray')
    plt.title(title2)

    plt.show()