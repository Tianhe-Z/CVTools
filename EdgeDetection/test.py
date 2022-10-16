import cv2
import numpy as np

import DataStructure

img = np.asarray(cv2.imread('.\\image\\lena.bmp') / 255.0)
# print(img.shape)
[h, w, d] = img.shape
img = img[0:h, 0:w, 1]

imgPyramid = DataStructure.Pyramid(img, 6)


