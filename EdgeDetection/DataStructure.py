import numpy as np

import Constant
import Tools


class Pyramid:
    def __init__(self, img, N):
        self.up = None
        self.down = None
        self.DoG_S = None
        self.DoG_O = None
        self.DoG = None
        self.img = img
        self.N = N
        self.GaussianPyramid = list()
        self.LaplacianPyramid = list()
        self.ImgPyramid = list()
        [self.h, self.w] = img.shape
        if type(img) is not np.ndarray:
            raise Exception("img format error")

    def basicpyramid(self):
        """
        这是pyramid类的基础方法，用于生成图像金字塔、基础的高斯金字塔和拉普拉斯金字塔

        :return:ImgPyramid, GaussianPyramid, LaplacianPyramid
        """
        self.ImgPyramid.append(self.img)
        for i in range(int(self.N)):
            self.down = Tools.convolution(Tools.downsampling(self.img), Constant.GaussianFilter256)
            self.ImgPyramid.append(self.down)
            self.up = Tools.convolution(Tools.upsampling(self.down), 4 * Constant.GaussianFilter256)
            self.GaussianPyramid.append(self.up)
            self.LaplacianPyramid.append(self.ImgPyramid[i] - self.GaussianPyramid[i])
            self.img = self.down
            [self.h, self.w] = self.img.shape
        return self.ImgPyramid, self.GaussianPyramid, self.LaplacianPyramid

    def differenceofgaussianpyramid(self):
        """
        让一个调用该方法的pyramid对象生成一个DoG金字塔，并且返回

        :return: DoG金字塔
        """
        [M, N] = self.img.shape
        Octave = np.log2(np.min([M, N])) - 3
        # 这里需要注意一下，我看知乎教程说是每组6层
        S = 6
        GaussianPyramid = list()
        sigma0 = 1.52
        OriginImg = Tools.gaussianfiltering(img=Tools.upsampling(self.img), sigma=sigma0, size=9, magnification=4)
        for i in range(int(Octave)):
            GaussianPyramid.append(list())
            for j in range(int(S)):
                sigma = sigma0 * (2 ** (i + j / 3))
                GaussianPyramid[i].append(Tools.gaussianfiltering(OriginImg, sigma, 9))
            OriginImg = Tools.downsampling(GaussianPyramid[i][3])
        self.DoG = list()
        self.DoG_O = int(Octave)
        self.DoG_S = int(S - 1)
        for i in range(self.DoG_O):
            self.DoG.append(list())
            for j in range(self.DoG_S):
                self.DoG[i].append(GaussianPyramid[i][j] - GaussianPyramid[i][j + 1])
        return self.DoG


class Conv2D:
    pass
