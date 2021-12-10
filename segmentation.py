import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    plt.show()
    return fig, ax

from skimage import data
import numpy as np
import matplotlib.pyplot as plt

image = data.astronaut()
# plt.imshow(image)
# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
# image_gray = color.rgb2gray(image)
image_gray = cv2.imread('../data/CC Lake extracted/2/DSC00899.jpg', 0)
# image_gray = cv2.resize(image_gray, None, fx = 0.125, fy = 0.125)
# image_show(image_gray)
print(image_gray.shape)
m, n = image_gray.shape

s = np.linspace(0, 2*np.pi, 500)
r = m//2 + m//2*np.sin(s)
c = n//2 + n//2*np.cos(s)
init = np.array([r, c]).T

# try parameters
a_range = [0.001, 0.01, 0.1, 0.2, 0.5, 1]
b_range = [0.01, 0.1, 1,5, 10, 20]
g_range = [0.001, 0.01, 0.1, 0.5, 1]
for a in a_range:
    for b in b_range:
        for g in g_range:

            snake = active_contour(gaussian(image_gray, 3, preserve_range=False),
                                   init, alpha=a, beta=b, gamma=g)
#
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(image_gray, cmap=plt.cm.gray)
            ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
            ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
            ax.set_xticks([]), ax.set_yticks([])
            ax.axis([0, image_gray.shape[1], image_gray.shape[0], 0])
            fname = '../data/contour/' + str(a) + '_' + str(b) + '_' + str(g) + '.png'
            plt.savefig(fname)


# cv2.imshow('a', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# snake = seg.active_contour(image_gray, points, alpha=0.06, beta=0.3)
# # print(snake)
# for i in snake:
#     # print(i)
#     image_out = cv2.circle(image_gray, (int(i[0]), int(i[1])), radius=1, color=(0), thickness=-1)

# import numpy as np
# import matplotlib.pyplot as plt
#
# from skimage.segmentation import random_walker
# from skimage.data import binary_blobs
# from skimage.exposure import rescale_intensity
# import skimage
#
# rng = np.random.default_rng()
#
# # Generate noisy synthetic data
# data = skimage.img_as_float(binary_blobs(length=128, seed=1))
# sigma = 0.35
# data += rng.normal(loc=0, scale=sigma, size=data.shape)
# data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
#                          out_range=(-1, 1))
#
# # The range of the binary image spans over (-1, 1).
# # We choose the hottest and the coldest pixels as markers.
# markers = np.zeros(data.shape, dtype=np.uint)
# markers[data < -0.95] = 1
# markers[data > 0.95] = 2
#
# # Run random walker algorithm
# labels = random_walker(data, markers, beta=10, mode='bf')
#
# # Plot results
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
#                                     sharex=True, sharey=True)
# ax1.imshow(data, cmap='gray')
# ax1.axis('off')
# ax1.set_title('Noisy data')
# ax2.imshow(markers, cmap='gray')
# ax2.axis('off')
# ax2.set_title('Markers')
# ax3.imshow(labels, cmap='gray')
# ax3.axis('off')
# ax3.set_title('Segmentation')
#
# fig.tight_layout()
# plt.show()
#
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.color import rgb2gray
# from skimage import data
# from skimage.filters import gaussian
# from skimage.segmentation import active_contour
#
#
# img = data.astronaut()
# img = rgb2gray(img)
#
# s = np.linspace(0, 2*np.pi, 400)
# r = 100 + 100*np.sin(s)
# c = 220 + 100*np.cos(s)
# init = np.array([r, c]).T
#
# snake = active_contour(gaussian(img, 3, preserve_range=False),
#                        init, alpha=0.015, beta=10, gamma=0.001)
#
# fig, ax = plt.subplots(figsize=(7, 7))
# ax.imshow(img, cmap=plt.cm.gray)
# ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
# ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])
#
# plt.show()

# fig, ax = image_show(image)
# ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
# ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);
