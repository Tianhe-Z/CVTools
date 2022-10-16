import cv2
import numpy as np

import Constant
import Tools

img = np.asarray(cv2.imread('.\\image\\1.png') / 255.0)
# print(img.shape)
[h, w, d] = img.shape
img = img[0:h, 0:w, 1]
# print(img.shape)
#
img = Tools.convolution(img, Constant.GaussianFilter)
# Gn, Gt = Tools.sobelmake(img)
# Gn = Tools.nms(Gn, Gt)
cv2.imshow('img', img)
cv2.waitKey(0)
