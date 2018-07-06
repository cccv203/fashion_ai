from nets.pyr_prm import Residual as pyr_res
from nets.pyr_prm import Res as res
import torch
import torch.nn as nn
import numpy as np
from code.config import cfg

class Lin(nn.Module):
    def __init__(self, numIn=128, numout=15):
        super(Lin, self).__init__()
        self.conv = nn.Conv2d(numIn, numout, 1)
        self.bn = nn.BatchNorm2d(numout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class HourGlass(nn.Module):

    def __init__(self, n=4, f=256, nModules=1, inputRes=256, type='no_preact', B=6, C=30):

        super(HourGlass, self).__init__()
        self.nModules = nModules
        self.inputRes = inputRes
        self.type = type
        self.B = B
        self.C = C
        self.n = n
        self.f = f

        self._init_layers(self.n, self.f,self.inputRes,self.nModules)

    def _init_layers(self, n, f, inputRes,nModules):
        # 上分支
        for i in range(nModules):
            if n >= 2:
                setattr(self, 'ResidualUp' + str(n) + '_1'+str(i), pyr_res(f,f,1,self.type,False,inputRes,self.B,self.C))
            else:
                setattr(self, 'ResidualUp' + str(n) + '_1'+str(i), res(f,f,1,self.type,False))
            if n >= 3:
                setattr(self, 'ResidualDown' + str(n) + '_2'+str(i), pyr_res(f,f,1,self.type,False,inputRes/2,self.B,self.C))
                setattr(self, 'ResidualDown' + str(n) + '_3'+str(i), pyr_res(f, f, 1, self.type, False, inputRes / 2, self.B, self.C))
                setattr(self, 'ResidualDown' + str(n) + '_4'+str(i), pyr_res(f, f, 1, self.type, False, inputRes / 2, self.B, self.C))
            else:
                setattr(self, 'ResidualDown' + str(n) + '_2'+str(i), res(f,f,1,self.type,False))
                setattr(self, 'ResidualDown' + str(n) + '_3'+str(i), res(f, f, 1, self.type, False))
                setattr(self, 'ResidualDown' + str(n) + '_4'+str(i), res(f, f, 1, self.type, False))
            # 下分支
        setattr(self, 'pool' + str(n) + '_1', nn.MaxPool2d(2, 2))
        if n > 1:
            self._init_layers(n - 1, f, inputRes/2,nModules)
        setattr(self, 'ResidualDown' + str(n) + '_3', res(f,f,1,self.type,False))
        setattr(self, 'SUSN' + str(n), nn.Upsample(scale_factor=2,mode='nearest'))

    def _forward(self, x, n, f):
        up1 = x
        for i in range(self.nModules):
            up1 = eval('self.ResidualUp' + str(n) + '_1'+str(i))(up1)
        low1 = eval('self.pool' + str(n) + '_1')(x)
        for i in range(self.nModules):
            low1 = eval('self.ResidualDown' + str(n) + '_2'+str(i))(low1)
        if n > 1:
            low2 = self._forward(low1,n-1,f)
        else:
            low2 = low1
            for i in range(self.nModules):
                low2 = eval('self.ResidualDown' + str(n) + '_3'+str(i))(low2)
        low3 = low2
        for i in range(self.nModules):
            low3 = eval('self.ResidualDown' + str(n) + '_4'+str(i))(low3)
        up2 = eval('self.SUSN' + str(n)).forward(low3)

        return up1 + up2

    def forward(self, x):
        return self._forward(x, self.n, self.f)

class preact(nn.Module):
    def __init__(self, numout):
        super(preact, self).__init__()
        self.bn = nn.BatchNorm2d(numout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(x))


class PyrNet(nn.Module):

    def __init__(self,cfg): ####更换imgsize只需更改inputRes
        """
        输入： 256^2
        """
        super(PyrNet,self).__init__()
        self.baseWidth = cfg.pyr['baseWidth']
        self.cardinality = cfg.pyr['cardinality']
        self.inputRes = np.array([int(res/4) for res in cfg.train_img_size])  #                 int(inputRes/4)   #64   = 512/8
        self.nStack = cfg.pyr['nStack']
        self.nFeats = cfg.pyr['nFeats']
        self.nClasses = cfg.joints_num
        self.nResidual = cfg.pyr['nResidual']

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.r1 = pyr_res(64,128,1,'no_preact',False,self.inputRes*2,self.baseWidth,self.cardinality)
        self.pool = nn.MaxPool2d(2,2)
        self.r4 = pyr_res(128,128,1,'preact',False,self.inputRes,self.baseWidth,self.cardinality)
        self.r5 = pyr_res(128,int(self.nFeats),1,'preact',False,self.inputRes,self.baseWidth,self.cardinality)

        self._init_stacked_hourglass()

    def _init_stacked_hourglass(self):
        for i in range(self.nStack):
            setattr(self,'hg'+str(i),HourGlass(4,self.nFeats,self.nResidual,self.inputRes,'no_preact',self.baseWidth,self.cardinality))
            setattr(self,'hg'+str(i)+'_preact',preact(self.nFeats))
            setattr(self,'hg'+str(i)+'_lin1',Lin(self.nFeats,self.nFeats))
            setattr(self,'hg'+str(i)+'_conv_pred',nn.Conv2d(self.nFeats,self.nClasses,1))
            if i < self.nStack - 1:
                setattr(self,'hg'+str(i)+'_conv1',nn.Conv2d(self.nFeats,self.nFeats,1))
                setattr(self, 'hg' + str(i) + '_conv2', nn.Conv2d(self.nClasses, self.nFeats, 1))


    def forward(self, input):
        x_1 = self.relu1(self.bn1(self.conv1(input)))
        x = self.r1(x_1)
        x = self.pool(x)
        x = self.r4(x)
        x = self.r5(x)
        out = []
        inter = x
        for i in range(self.nStack):
            hg = eval('self.hg' + str(i))(inter)
            ll = eval('self.hg' + str(i)+'_preact')(hg)
            ll = eval('self.hg' + str(i)+'_lin1')(ll)
            tmpOut = eval('self.hg' + str(i) + '_conv_pred')(ll)
            out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = eval('self.hg' + str(i) + '_conv1')(ll)
                tmpOut_ = eval('self.hg' + str(i) + '_conv2')(tmpOut)
                inter = inter + ll_ + tmpOut_
        return torch.stack(out[:-1],dim=1),out[-1]

if __name__ == '__main__':
    net = PyrNet(cfg).cuda()
    x = torch.autograd.Variable(torch.randn((2,3,512,512))).cuda()
    y,yy = net(x)
    print(y.size())
    print(yy.size())
