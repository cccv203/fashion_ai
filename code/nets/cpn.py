import torch.nn as nn
import torch
import torch.nn.functional as F
from nets.resnet import (resnet50,resnet101,Bottleneck)
from code.config import cfg
from torch.autograd import Variable


class CPN(nn.Module):
    def __init__(self, cfg):
        super(CPN, self).__init__()
        self.cfg = cfg
        self._resnet_init_modules()
        self._init_modules()
        self._init_weights()

    def _upsample_add(self,x,y):
        B,C,H,W = x.size()
        return F.upsample(y, size=(H,W), mode='bilinear') + x


    def _resnet_init_modules(self):
        if self.cfg.cpn['basenet'] == 'resnet50':
            resnet = resnet50(self.cfg.basenet_pretrained)
        else:
            resnet = resnet101(self.cfg.basenet_pretrained)

        self.RCNN_layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.RCNN_layer1 = nn.Sequential(resnet.layer1)
        self.RCNN_layer2 = nn.Sequential(resnet.layer2)
        self.RCNN_layer3 = nn.Sequential(resnet.layer3)
        self.RCNN_layer4 = nn.Sequential(resnet.layer4)

        # Fix blocks
        for p in self.RCNN_layer0[0].parameters(): p.requires_grad = False
        for p in self.RCNN_layer0[1].parameters(): p.requires_grad = False

        assert (0 <= self.cfg.RESNET_FIXED_BLOCKS < 4)
        if self.cfg.RESNET_FIXED_BLOCKS >= 3:
            for p in self.RCNN_layer3.parameters(): p.requires_grad = False
        if self.cfg.RESNET_FIXED_BLOCKS >= 2:
            for p in self.RCNN_layer2.parameters(): p.requires_grad = False
        if self.cfg.RESNET_FIXED_BLOCKS >= 1:
            for p in self.RCNN_layer1.parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                # pass
                for p in m.parameters(): p.requires_grad = False

        self.RCNN_layer0.apply(set_bn_fix)
        self.RCNN_layer1.apply(set_bn_fix)
        self.RCNN_layer2.apply(set_bn_fix)
        self.RCNN_layer3.apply(set_bn_fix)
        self.RCNN_layer4.apply(set_bn_fix)

    def _init_modules(self):

        for i in range(5, 1, -1):
            setattr(self, 'global_' + str(i) + '_reduce_dim', nn.Sequential(nn.Conv2d(int(2048 / (2 ** (5 - i))),
                                                                 256, kernel_size=1, stride=1, padding=0),
                                                                            nn.ReLU(inplace=True)))

            setattr(self, 'global_' + str(i) + '_heatmap',nn.Sequential(
                nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.cfg.joints_num, kernel_size=3, stride=1, padding=1),
                nn.Upsample(size=self.cfg.hm_size, mode='bilinear')
            ))
            if i<5:
                setattr(self, 'global_' + str(i) + '_upConv', nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                                                            ))

        for i in range(4):
            for j in range(i):
                setattr(self, 'refine_' + str(i+2)+'_'+str(j)+'_bottle', Bottleneck(256, 64, 1))
            setattr(self, 'refine_' + str(i) + '_upsample', nn.Upsample(size=self.cfg.hm_size, mode='bilinear'))
        self.refine_hm_bottle = Bottleneck(256*4, 64*4, 1)
        self.refine_hms = nn.Conv2d(256*4,self.cfg.joints_num,kernel_size=3,padding=1)

    def _init_weights(self):
        def _init_weight_sequence(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.normal(m.weight, mean=0, std=0.0001)

        for name in self.__dict__['_modules']:
            if 'global' in name or 'refine' in name:
                classname = self.__dict__['_modules'][name]
                classname.apply(_init_weight_sequence)

    def _global(self, inputs):
        c1 = self.RCNN_layer0(inputs)
        c2 = self.RCNN_layer1(c1)  # 256*64*64
        c3 = self.RCNN_layer2(c2)  # 512*32*32
        c4 = self.RCNN_layer3(c3)  # 1024*16*16
        c5 = self.RCNN_layer4(c4)  # 2048*16*16
        output = [None] * 4
        global_hms = []
        last_fm=None
        for i in range(5, 1, -1):
            p = eval('self.global_' + str(i) + '_reduce_dim')(eval('c' + str(i)))
            if last_fm is not None:
                up = self._upsample_add(p, last_fm)
                up = eval('self.global_' + str(i) + '_upConv')(up)
                last_fm = p + up
            else:
                last_fm = p

            global_hms.append(p)
            ll = eval('self.global_' + str(i) + '_heatmap')(p)
            output[5-i] = ll
        global_hms.reverse()
        return output,global_hms

    def _refine(self,global_hms):
        refine_hms = []
        for i,hm in enumerate(global_hms):
            for j in range(i):
                hm = eval('self.refine_' + str(i+2)+'_'+str(j)+'_bottle')(hm)
            hm = eval('self.refine_' + str(i)+ '_upsample')(hm)
            refine_hms.append(hm)
        refine_hm = torch.cat(refine_hms,dim=1)
        refine_hm = self.refine_hm_bottle(refine_hm)
        refine_hm = self.refine_hms(refine_hm)
        return refine_hm

    def forward(self, input):
        global_output,gloal_hms = self._global(input)
        refine_output = self._refine(gloal_hms)
        return torch.stack(global_output, dim=1),refine_output


if __name__ == '__main__':

    torch.manual_seed(1)
    net = CPN(cfg).cuda()
    global_output, refine_output = net(Variable(torch.randn(1, 3, 384, 256)).cuda())
    print(global_output.size())
    print(refine_output.size())
