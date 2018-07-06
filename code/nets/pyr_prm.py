import torch.nn as nn
import math
import torch


class Pyramid(nn.Module):
    def __init__(self, D, C, inputRes):
        super(Pyramid, self).__init__()
        self.C = C
        self.sc = 2 ** (1 / C)
        for i in range(self.C):
            scaled = 1 / self.sc ** (i + 1)
            setattr(self, 'conv_' + str(i), nn.Conv2d(D, D, kernel_size=2, stride=1, padding=1))
            setattr(self, 'SpatialFractionalMaxPooling_' + str(i),
                    nn.FractionalMaxPool2d(kernel_size=2, output_ratio=scaled))
            setattr(self, 'SpatialUpSamplingBilinear_' + str(i),
                    nn.Upsample(size=(int(inputRes[0]), int(inputRes[1])), mode='bilinear'))

    def forward(self, inputs):
        pyr = eval('self.SpatialFractionalMaxPooling_' + str(0))(inputs)
        pyr = eval('self.conv_' + str(0))(pyr)
        pyr = eval('self.SpatialUpSamplingBilinear_' + str(0))(pyr)
        for i in range(1, self.C):
            x = eval('self.SpatialFractionalMaxPooling_' + str(i))(inputs)
            x = eval('self.conv_' + str(i))(x)
            x = eval('self.SpatialUpSamplingBilinear_' + str(i))(x)
            pyr += x
        return pyr


class convBlock(nn.Module):

    def __init__(self, numIn, numOut, inputRes, type, baseWidth, cardinality, stride):
        super(convBlock, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.inputRes = inputRes
        self.type = type
        self.baseWidth = baseWidth
        self.cardinality = cardinality
        self.stride = stride

        self.D = math.floor(self.numOut / self.baseWidth)
        self.C = self.cardinality

        self.batch = nn.BatchNorm2d(self.numIn)
        self.relu_bat = nn.ReLU(inplace=True)

        self.s1 = nn.Sequential(
            nn.Conv2d(self.numIn, int(self.numOut / 2), kernel_size=1, stride=1),
            nn.BatchNorm2d(int(self.numOut / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(self.numOut / 2), int(self.numOut / 2), kernel_size=3, stride=self.stride, padding=1),
        )

        self.batch_s2_1 = nn.BatchNorm2d(self.numIn)
        self.relu_s2_1 = nn.ReLU(inplace=True)
        nn.ReLU(inplace=True),
        self.s2 = nn.Sequential(
            nn.Conv2d(self.numIn, self.D, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(self.D),
            nn.ReLU(inplace=True),
            Pyramid(self.D, self.C, self.inputRes),
            nn.BatchNorm2d(self.D),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.D, int(self.numOut / 2), kernel_size=1),
        )

        self.batch_x = nn.BatchNorm2d(int(self.numOut / 2))
        self.relu_x = nn.ReLU(inplace=True)
        self.conv_x = nn.Conv2d(int(self.numOut / 2), self.numOut, kernel_size=1)

    def _s1(self, inputs):
        if self.type != 'no_preact':
            inputs = self.batch(inputs)
            inputs = self.relu_bat(inputs)
        x = self.s1(inputs)
        return x

    def _s2(self, inputs):
        if self.type != 'no_preact':
            inputs = self.batch_s2_1(inputs)
            inputs = self.relu_s2_1(inputs)
        x = self.s2(inputs)
        return x

    def forward(self, inputs):
        s1 = self._s1(inputs)  #不改变尺寸
        s2 = self._s2(inputs)  #inputRes
        x = s1 + s2
        x = self.batch_x(x)
        x = self.relu_x(x)
        x = self.conv_x(x)
        return x




class conv_block(nn.Module):

    def __init__(self,numIn,numOut,stride,type):
        super(conv_block, self).__init__()
        self.type = type
        self.s1 = nn.Sequential(
            nn.BatchNorm2d(numIn),
            nn.ReLU(inplace=True),
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(numIn,int(numOut/2),kernel_size=1),
            nn.BatchNorm2d(int(numOut/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(numOut/2),int(numOut/2),kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(int(numOut/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(numOut/2),numOut,kernel_size=1),
        )

    def forward(self, inputs):
        if self.type != 'no_preact':
            inputs = self.s1(inputs)
        x = self.s2(inputs)
        return x

class skipLayer(nn.Module):
    def __init__(self, numIn, numOut, stride, useConv):
        super(skipLayer, self).__init__()

        self.numIn = numIn
        self.numOut = numOut
        self.stride = stride
        self.useConv = useConv

        self.s1 = nn.Sequential(
            nn.BatchNorm2d(self.numIn),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.numIn, self.numOut, kernel_size=1, stride=self.stride)
        )

        self.s2 = nn.Sequential(
            nn.BatchNorm2d(self.numIn),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.numIn, self.numOut, kernel_size=1, stride=self.stride)
        )

    def forward(self, inputs):
        if self.useConv:
            x = self.s1(inputs)
            return x
        elif (self.numIn == self.numOut) & (self.stride == 1):
            return inputs
        else:
            x = self.s2(inputs)
            return x


class Residual(nn.Module):
    def __init__(self, numIn, numOut, stride=1, type='no_preact', useConv=False, inputRes=(24,16), baseWidth=6, cardinality=4):
        super(Residual, self).__init__()

        self.convBlock = convBlock(numIn, numOut, inputRes, type, baseWidth, cardinality, stride)
        self.skipLayer = skipLayer(numIn, numOut, stride, useConv)

    def forward(self, inputs):
        convBlock = self.convBlock(inputs)
        skipLayer = self.skipLayer(inputs)
        return convBlock + skipLayer


class Res(nn.Module):
    def __init__(self, numIn, numOut, stride, type, useConv):
        super(Res, self).__init__()

        self.convBlock = conv_block(numIn, numOut, stride, type)
        self.skipLayer = skipLayer(numIn, numOut, stride, useConv)

    def forward(self, inputs):
        convBlock = self.convBlock(inputs)
        skipLayer = self.skipLayer(inputs)
        return convBlock + skipLayer



if __name__ == '__main__':
    res = Residual(3, 128, 1, 'preact', False, (386,256), 6, 30)
    x = torch.autograd.Variable(torch.randn((1, 3, 386, 256)))
    print(res(x))
