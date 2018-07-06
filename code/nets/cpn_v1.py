import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from nets.resnet import (resnet50,resnet101,Bottleneck)
from nets.pyr_prm import Residual
from code.config import cfg
from torch.autograd import Variable


class CPN(nn.Module):
    def __init__(self, cfg):
        super(CPN, self).__init__()
        self.cfg = cfg
        self._init_modules()
        self._init_weights()

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_layer0.train()
            self.RCNN_layer1.train()
            self.RCNN_layer2.train()
            self.RCNN_layer3.train()
            self.RCNN_layer4.train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    # m.eval()
                    pass

            self.RCNN_layer0.apply(set_bn_eval)
            self.RCNN_layer1.apply(set_bn_eval)
            self.RCNN_layer2.apply(set_bn_eval)
            self.RCNN_layer3.apply(set_bn_eval)
            self.RCNN_layer4.apply(set_bn_eval)

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
                pass
                # for p in m.parameters(): p.requires_grad = False

        self.RCNN_layer0.apply(set_bn_fix)
        self.RCNN_layer1.apply(set_bn_fix)
        self.RCNN_layer2.apply(set_bn_fix)
        self.RCNN_layer3.apply(set_bn_fix)
        self.RCNN_layer4.apply(set_bn_fix)

    def _build_global_out(self):
        for i in range(5, 1, -1):
            setattr(self, 'pyramid_' + str(i) + '_reduce_dim', nn.Sequential(nn.Conv2d(int(2048 / (2 ** (5 - i))),
                                                                                       self.cfg.base_fea_n,
                                                                                       kernel_size=1,
                                                                                       stride=1, padding=0),
                                                                             nn.BatchNorm2d(self.cfg.base_fea_n),
                                                                             nn.ReLU(inplace=True)))
            setattr(self, 'pyramid_' + str(i) + '_fuse', nn.Sequential(
                nn.Conv2d(self.cfg.base_fea_n, self.cfg.base_fea_n, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.cfg.base_fea_n),
                nn.ReLU(inplace=True)))
            setattr(self, 'pyramid_' + str(i) + '_dila_fuse', nn.Sequential(
                nn.Conv2d(self.cfg.base_fea_n, self.cfg.base_fea_n, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm2d(self.cfg.base_fea_n),
                nn.ReLU(inplace=True)))
            setattr(self, 'gout_' + str(i) + '_heatmap',
                    nn.Conv2d(256, self.cfg.joints_num, kernel_size=3, stride=1, padding=1))
            setattr(self, 'gout_' + str(i) + '_upsample',
                    nn.Upsample(size=self.cfg.hm_size, mode='bilinear'))
            if i > 2:
                setattr(self, 'pyramid_' + str(i) + '_up_and', nn.Sequential(
                    nn.Conv2d(self.cfg.base_fea_n, self.cfg.base_fea_n, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(self.cfg.base_fea_n),
                    nn.ReLU(inplace=True)))
                setattr(self, 'global_' + str(i) + '_increase_dim', nn.Sequential(
                    nn.Conv2d(self.cfg.joints_num, self.cfg.base_fea_n, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(self.cfg.base_fea_n),
                    nn.ReLU(inplace=True)))

    def _build_refine_out(self):
        for i in range(2, 6):
            setattr(self, 'refine_' + str(i) + '_dila1', nn.Sequential(
                nn.Conv2d(self.cfg.base_fea_n, self.cfg.base_fea_n, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm2d(self.cfg.base_fea_n),
                nn.ReLU(inplace=True)))
            setattr(self, 'refine_' + str(i) + '_dila2', nn.Sequential(
                nn.Conv2d(self.cfg.base_fea_n, self.cfg.base_fea_n, kernel_size=3, padding=3, dilation=3),
                nn.BatchNorm2d(self.cfg.base_fea_n),
                nn.ReLU(inplace=True)))
            for j in range(i - 2):
                inputRes = np.array(self.cfg.hm_size) / 2 if i == 3 else np.array(self.cfg.hm_size) / 4
                if self.cfg.cpn['bottleneck'] == 'resnet':
                    setattr(self, 'refine_' + str(i) + '_' + str(j) + '_bottleneck',Bottleneck(self.cfg.base_fea_n,
                                                                                               int(self.cfg.base_fea_n/4), 1))
                else:
                    setattr(self, 'refine_' + str(i) + '_' + str(j) + '_bottleneck',
                            Residual(self.cfg.base_fea_n, self.cfg.base_fea_n, 1, inputRes=inputRes))

        for i in range(2, 6):
            setattr(self, 'refine_' + str(i) + '_upsample',
                    nn.Upsample(size=self.cfg.hm_size[0], mode='bilinear'))
        if self.cfg.cpn['bottleneck'] == 'resnet':
            self.refine_hm_bottle = Bottleneck(self.cfg.base_fea_n*4,self.cfg.base_fea_n,1)
        else:
            self.refine_hm_bottle = Residual(self.cfg.base_fea_n * 4, self.cfg.base_fea_n * 4, 1, inputRes=self.cfg.hm_size)
        self.refine_out = nn.Conv2d(self.cfg.base_fea_n * 4, self.cfg.joints_num, kernel_size=(3, 3), stride=1,
                                    padding=1)

    def _init_modules(self):
        self._resnet_init_modules()
        self._build_global_out()
        self._build_refine_out()

    def _init_weights(self):
        def _init_weight_sequence(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.normal(m.weight, mean=0, std=0.0001)

        for name in self.__dict__['_modules']:
            if 'pyramid' in name or 'global' in name or 'refine' in name or 'gout' in name:
                classname = self.__dict__['_modules'][name]
                classname.apply(_init_weight_sequence)

    def _upsample_add(self, x, y, z):
        _, _, H, W = x.size()
        return F.upsample(y, size=(H, W), mode='bilinear') + F.upsample(z, size=(H, W), mode='bilinear') + x


    def _g_out(self, input):
        c1 = self.RCNN_layer0(input)
        c2 = self.RCNN_layer1(c1)  # 256*64*64
        c3 = self.RCNN_layer2(c2)  # 512*32*32
        c4 = self.RCNN_layer3(c3)  # 1024*16*16
        c5 = self.RCNN_layer4(c4)  # 2048*16*16

        g_out = []
        for_refine = []
        for i in range(5, 1, -1):
            p = eval('self.pyramid_' + str(i) + '_reduce_dim')(eval('c' + str(i)))
            if i < 5:
                p = self._upsample_add(p, up, ll_)
            p_fuse = eval('self.pyramid_' + str(i) + '_fuse')(p)
            p_dila_fue = eval('self.pyramid_' + str(i) + '_dila_fuse')(p)
            p = p_dila_fue + p_fuse
            up = eval('self.pyramid_' + str(i) + '_up_and')(p) if i > 2 else None

            for_refine.append(p)
            ll = eval('self.gout_' + str(i) + '_heatmap')(p)
            ll_ = eval('self.global_' + str(i) + '_increase_dim')(ll) if i > 2 else None
            g_out.append(eval('self.gout_' + str(i) + '_upsample')(ll))
        for_refine.reverse()
        return g_out, for_refine

    def _r_out(self, for_refine):
        refine_hms = []
        for i, hm in enumerate(for_refine):
            hm_dila1 = eval('self.refine_' + str(i + 2) + '_dila1')(hm)
            hm_dila2 = eval('self.refine_' + str(i + 2) + '_dila2')(hm)
            hm = hm_dila1 + hm_dila2
            for j in range(i):
                hm = eval('self.refine_' + str(i + 2) + '_' + str(j) + '_bottleneck')(hm)
            hm = eval('self.refine_' + str(i + 2) + '_upsample')(hm)
            refine_hms.append(hm)
        refine_hm = torch.cat(refine_hms, dim=1)
        refine_hm = self.refine_hm_bottle(refine_hm)
        refine_hm = self.refine_out(refine_hm)
        return refine_hm

    def forward(self, input):

        global_output, for_refine = self._g_out(input)
        refine_output = self._r_out(for_refine)
        return torch.stack(global_output, dim=1), refine_output


if __name__ == '__main__':
    torch.manual_seed(1)
    net = CPN(cfg)
    print(net)
