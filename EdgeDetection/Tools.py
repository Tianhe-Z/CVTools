import math

import numpy as np

import Constant


def convolution(img, kernel):
    """
    对输入图像使用输入的核进行卷积

    :param img:待卷积图像
    :param kernel: 卷积核，尺寸需要为奇数
    :return: 卷积运算后的图像
    """
    if type(img) is np.ndarray and type(img) is np.ndarray:
        [imgh, imgw] = img.shape
        [kernelh, kernelw] = kernel.shape
        if kernelh % 2 == 0 or kernelw % 2 == 0:
            raise Exception("kernel size is not odd")
        padding_img = np.zeros([imgh + (kernelh - 1), imgw + (kernelw - 1)], np.float32)
        rs = np.zeros([imgh, imgw], np.float32)
        padding_img[int((kernelh - 1) / 2):int((kernelh - 1) / 2) + imgh,
        int((kernelw - 1) / 2):int((kernelw - 1) / 2) + imgw] = img
        for i in range(int((kernelh - 1) / 2), int((kernelh - 1) / 2 + imgh)):
            for j in range(int((kernelw - 1) / 2), int((kernelw - 1) / 2 + imgw)):
                roi = padding_img[i - int((kernelh - 1) / 2):i + int((kernelh - 1) / 2) + 1,
                      j - int((kernelw - 1) / 2):j + int((kernelw - 1) / 2) + 1]
                rs[i - int((kernelh - 1) / 2)][j - int((kernelw - 1) / 2)] = np.sum(roi * kernel)
        return rs
    else:
        raise Exception("img or kernel isn't numpy array")


def upsampling(img):
    """
    对输入图像进行上采样，没有进行高斯模糊

    :param img:需要上采样的图像
    :return: 进行上采样后的图像
    """
    if type(img) is not np.ndarray:
        raise Exception("image format error")
    [h, w] = img.shape
    for i in range(h):
        img = np.insert(img, h - i, [0] * w, axis=0)
    for j in range(w):
        img = np.insert(img, w - j, [0] * h * 2, axis=1)
    # img = convolution(img, 4 * Constant.GaussianFilter256)
    return img


def downsampling(img):
    """
    对输入图像进行下采样，没有进行高斯模糊

    :param img: 需要进行下采样的图像
    :return: 进行下采样后的图像
    """
    if type(img) is not np.ndarray:
        raise Exception("image format error")
    # img = convolution(img, Constant.GaussianFilter256)
    [h, w] = img.shape
    deleteRow = list(range(0, h, 2))
    deleteCol = list(range(0, w, 2))
    img = np.delete(img, deleteRow, axis=0)
    img = np.delete(img, deleteCol, axis=1)
    return img


def elementmapping(a):
    """
    简单来说，这是一个屎山，无需人工调用

    :param a:
    :return:
    """
    b = np.rad2deg(a)
    dif = 0.0
    if -22.5 <= b < 22.5:
        dif = abs(b - 0.0)
        b = 0.0
    elif 22.5 <= b < 67.5:
        dif = abs(b - 45.0)
        b = 45.0
    elif 67.5 <= b < 112.5:
        dif = abs(b - 90.0)
        b = 90.0
    elif 112.5 <= b < 157.5:
        dif = abs(b - 135.0)
        b = 135.0
    elif 157.5 <= b <= 180:
        dif = abs(b - 180.0)
        b = 180.0
    elif -22.5 > b >= -67.5:
        dif = abs(b + 45.0)
        b = -45.0
    elif -67.5 > b >= -112.5:
        dif = abs(b + 90.0)
        b = -90.0
    elif -112.5 > b >= -157.5:
        dif = abs(b + 135.0)
        b = -135.0
    elif -157.5 > b >= -180:
        dif = abs(b + 180.0)
        b = -180.0
    dif = np.deg2rad(dif)
    b = abs(b)
    return b, dif


def gradientmapping(Gn, Gt):
    """
    说明：这个函数是对求出的图像梯度图进行映射的，将梯度角度映射到规定的角度，方便后续NMS运行

    :param Gn: 待映射的梯度图
    :param Gt: 梯度的方向图
    :return: Gn，Gt映射完成的梯度图和方向图
    """
    if type(Gn) is not np.ndarray or type(Gt) is not np.ndarray:
        raise Exception('Gn or Gt format error')
    [h, w] = Gn.shape
    mapping = np.vectorize(elementmapping)
    Gtmp = mapping(Gt)
    Gt, Gdif = Gtmp[0], Gtmp[1]
    for i in range(h):
        for j in range(w):
            Gn[i][j] = math.cos(Gdif[i][j]) * Gn[i][j]
    return Gn, Gt


def sobelmake(img):
    """
    对输入图像使用sobel算子进行卷积，同时映射梯度图

    :param img:待求梯度的图像
    :return: Gn，每个像素的梯度大小
             Gt，每个像素的梯度方向
    """
    if type(img) is not np.ndarray:
        raise Exception('image format error')
    [h, w] = img.shape
    Gx = convolution(img, Constant.SobelXFilter)
    Gy = convolution(img, Constant.SobelYFilter)
    Gn = np.zeros([h, w], np.float32)
    Gt = np.zeros([h, w], np.float32)
    for i in range(h):
        for j in range(w):
            Gn[i][j] = math.sqrt(Gx[i][j] ** 2 + Gy[i][j] ** 2)
            Gt[i][j] = math.atan(Gy[i][j] / Gx[i][j])
    Gn, Gt = gradientmapping(Gn, Gt)
    return Gn, Gt


def prewittmake(img):
    """
    对输入图像使用prewitt算子进行卷积，同时映射梯度图

    :param img:待求梯度的图像
    :return: Gn，每个像素的梯度大小,Gt，每个像素的梯度方向
    """
    if type(img) is not np.ndarray:
        raise Exception('image format error')
    [h, w] = img.shape
    Gx = convolution(img, Constant.PrewittXFilter)
    Gy = convolution(img, Constant.PrewittYFilter)
    Gn = np.zeros([h, w], np.float32)
    Gt = np.zeros([h, w], np.float32)
    for i in range(h):
        for j in range(w):
            Gn[i][j] = math.sqrt(Gx[i][j] ** 2 + Gy[i][j] ** 2)
            Gt[i][j] = math.atan(Gy[i][j] / Gx[i][j])
    Gn, Gt = gradientmapping(Gn, Gt)
    return Gn, Gt


def elementnms(x, dir):
    """
    说明：这也是一个屎山，此函数仅供非极大值抑制函数使用，无需主动调用

    :param x:
    :param dir:
    :return:
    """
    if type(x) is not np.ndarray or type(x) is not np.ndarray:
        raise Exception('x format error')
    if dir == 0.0:
        if x[0][1] <= x[1][1] and x[2][1] <= x[1][1]:
            x[1][1] = x[1][1]
        else:
            x[1][1] = 0.0
    elif dir == 45.0:
        if x[0][0] <= x[1][1] and x[2][2] <= x[1][1]:
            x[1][1] = x[1][1]
        else:
            x[1][1] = 0
    elif dir == 90.0:
        if x[0][1] <= x[1][1] and x[1][2] <= x[1][1]:
            x[1][1] = x[1][1]
        else:
            x[1][1] = 0
    elif dir == 135.0:
        if x[0][2] <= x[1][1] and x[2][0] <= x[1][1]:
            x[1][1] = x[1][1]
        else:
            x[1][1] = 0
    elif dir == 180.0:
        if x[0][1] <= x[1][1] and x[3][1] <= x[1][1]:
            x[1][1] = x[1][1]
        else:
            x[1][1] = 0.0
    else:
        return 0
    # print(dir)
    return x[1][1]


def nms(Gn, Gt):
    """
    说明：这个nms的效果好像太强劲了，抑制完的图像已经不剩下什么了

    :param Gn: 需要进行nms的梯度图
    :param Gt: 梯度图对应的方向图
    :return: nms完成后的图像
    """
    [h, w] = Gn.shape
    Gtmp = np.zeros([2 + h, 2 + w], np.float32)
    Gtmp[1:h + 1, 1:w + 1] = Gn
    for i in range(h):
        for j in range(w):
            Gn[i][j] = elementnms(Gtmp[i:i + 3, j:j + 3], Gt[i][j])
    return Gn


def gaussianfiltering(img, sigma, size, magnification=1):
    """
    该函数指定模糊系数sigma，以及高斯核大小size，输入需要高斯模糊的图像，返回高斯模糊后的图像

    :param magnification:高斯核乘以倍率
    :param img: 需要高斯模糊的图像
    :param sigma: 模糊系数
    :param size: 高斯核大小
    :return: 高斯模糊后的图像
    """
    gaussianKernel = generategaussiankernel(sigma, size)
    return convolution(img, magnification * gaussianKernel)


def generategaussiankernel(sigma, size):
    """
    输入sigma，size，返回一个高斯核

    :param sigma: 高斯函数的标准差
    :param size: 高斯核的大小
    :return: 高斯核
    """
    if size % 2 == 0:
        raise Exception("kernel size is not odd")
    X = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    Y = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    x, y = np.meshgrid(X, Y)
    tmp = 1 / (2 * np.pi * sigma ** 2) * np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))
    Z = tmp.sum()
    gaussianKernel = (1 / Z) * tmp
    return gaussianKernel
