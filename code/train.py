from nets.cpn import CPN
from nets.cpn_v1 import CPN as CPN_v1
from nets.pyrnet import PyrNet
from dataset.dataflow import DataProvider
from utils.loss import *
import  time
import numpy as np
import sys
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from  config import  cfg



def train(train_loader, model, optimizer,lr,epoch):
    epochStartTime = time.time()
    avgCost = 0.
    costNow = 0.
    costTotal = 0.0
    for i, (img,hm,r_hm,w,w2,joints) in enumerate(train_loader):
        inputs = Variable(img).cuda()
        g_gt = Variable(hm,requires_grad = False).cuda()
        r_gt = Variable(r_hm,requires_grad = False).cuda()

        global_weights = Variable(w).cuda()
        refine_weights = Variable(w2).cuda()

        optimizer.zero_grad()

        global_out, refine_out = model(inputs)

        loss_total = heatmap_loss(global_out, g_gt, refine_out, r_gt, global_weights, refine_weights,
                                  int(0.8 * cfg.joints_num), int(0.5 * cfg.joints_num),
                                  batch_size=cfg.batch_size, step=i + len(train_loader) * (epoch + 0))

        loss_total.backward()
        optimizer.step()

        percent = ((i + 1) / len(train_loader)) * 100
        num = np.int(20 * percent / 100)
        tToEpochEnd = int((time.time() - epochStartTime) * (100 - percent) / (percent))
        sys.stdout.write(
            '\r Progress: {0}>'.format(">" * num) + "{0}>".format(" " * (20 - num)) + '||' +
            str(percent)[:3] + '%' + '\t | loss_now: ' + str(costNow) +
            '\t | avg_loss: ' + str(avgCost) + '\t | time_to_epoch_end: ' + str(tToEpochEnd) + ' sec.' +
            '\t | lr: ' + str(lr) + ' | step: ' + str(i))
        sys.stdout.flush()
        costNow = loss_total.data[0]
        costTotal += costNow
        avgCost = costTotal / (i + 1)
        if i%100==1:
            save_test_img(inputs, refine_out, g_gt, i)

def main():
    if cfg.nets == 'cpn':
        model = CPN(cfg).cuda()
    elif cfg.nets == 'cpn_v1':
        model = CPN_v1(cfg).cuda()
    else:
        model = PyrNet(cfg).cuda()
    try:
        state_dict = torch.load('../logs/'+cfg.nets+'.pkl')
        model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})
        print('load model from ','mix_resnet101.pkl', ' over')
    except:
        print("no pretrained models, let's train from the beginning")
        pass
    model.train()
    lr = cfg.base_lr

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            params += [{'params': [value], 'lr': cfg.base_lr, 'weight_decay': cfg.weight_decay}]

    optimizer = torch.optim.Adam(params, lr=cfg.base_lr,weight_decay=cfg.weight_decay)
    train_loader = torch.utils.data.DataLoader(
        DataProvider(cfg),batch_size=cfg.batch_size, shuffle=True)
    for epoch in range(cfg.epoches):
        lr = adjust_learning_rate(optimizer, epoch, lr, [2, 4, 6, 8], 0.5)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
        train(train_loader, model, optimizer,lr,epoch)
        torch.save(model.state_dict(), '../logs/'+cfg.nets+'.pkl')


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def save_test_img(images, heat_map, gt_maps, idx, category='mix'):
    images = images[0].clone().cpu().data.numpy().transpose([1, 2, 0])
    img = images.copy()
    map = heat_map[0].clone().cpu().data.numpy()
    map = map.copy()
    gt_map = gt_maps[0,0].clone().cpu().data.numpy()
    gt_map = gt_map.copy()
    for i in range(map.shape[0]):
        yy_2, xx_2 = np.where(gt_map[i] == gt_map[i].max())
        if xx_2[0]!=0 and yy_2[0]!=0:
            cv2.circle(img, (int(xx_2[0] * 4), int(yy_2[0] * 4)), radius=1, thickness=2,
                       color=(255, 0, 0))

            yy_2, xx_2 = np.where(map[i] == map[i].max())
            cv2.circle(img, (int(xx_2[0] * 4+2), int(yy_2[0] * 4+2)), radius=1, thickness=2,
                       color=(0, 0, 255))
            cv2.putText(img, str(i), (int(xx_2[0] * 4), int(yy_2[0] * 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
    cv2.imwrite('../data/imgs/training_img/'+category + str(idx) + '.jpg', img)

if __name__ == '__main__':
    main()


