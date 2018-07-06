import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from utils.logger import Logger
logger = Logger('../logs')


def ohkm(loss_heatmap,refine_ohkm_n,batch_size):
    ohkm_total_loss = 0.
    for i in range(batch_size):
        sub_loss = loss_heatmap[i]
        ohkm_loss,_ = torch.sort(sub_loss,dim=0,descending=True)
        ohkm_total_loss += torch.mean(ohkm_loss[:refine_ohkm_n])
    return ohkm_total_loss/batch_size

def global_ohkm(loss_heatmap,global_ohkm_n,batch_size,channel):
    ohkm_total_loss = 0.
    for i in range(batch_size):
        for j in range(channel):
            sub_loss = loss_heatmap[i,j]
            ohkm_loss, _ = torch.sort(sub_loss, dim=0, descending=True)
            ohkm_total_loss += torch.mean(ohkm_loss[:global_ohkm_n])
    return ohkm_total_loss / (batch_size*channel)

def mask(weights,nn):
    N, C= weights.shape
    N_idx = []
    C_idx = []

    for n in range(N):
        ohkm_n = torch.sum(weights[n])
        for c in range(int(ohkm_n*nn)):
            N_idx.append(n)
            C_idx.append(c)
    mask = np.zeros(weights.shape)
    mask[N_idx, C_idx] = 1.
    mask = Variable(torch.from_numpy(mask).float()).cuda()
    return mask

def heatmap_loss(g_out,g_gt, r_out,r_gt,global_weights,refine_weights,global_ohkm_n, refine_ohkm_n,batch_size,step):
    b,c,j,h,w = g_gt.size()

    loss_heatmap = (nn.MSELoss(reduce=False)(g_out,g_gt)).view(b,c,j,-1)*(global_weights.unsqueeze(1).unsqueeze(3))
    global_ohkm_loss,_ = torch.sort(loss_heatmap,dim=2,descending=True)
    global_mask = mask(weights=global_weights,nn=0.8)
    global_ohkm_loss *= global_mask.unsqueeze(1).unsqueeze(3)
    global_ohkm_loss = torch.sum(global_ohkm_loss)/(c*w*h*torch.sum(global_mask))

    global_ohem_loss,_ = torch.sort(loss_heatmap,dim=0,descending=True)
    global_ohem_loss = torch.sum(global_ohem_loss[:int(batch_size * 0.8)])/(c*w*h*torch.sum(global_weights))

    loss_heatmap = (nn.MSELoss(reduce=False)(r_out,r_gt)).view(b,j,-1)*(refine_weights.unsqueeze(2))
    refine_ohkm_loss,_ = torch.sort(loss_heatmap,dim=1,descending=True)
    refine_mask = mask(weights=refine_weights,nn=0.5)
    refine_ohkm_loss *= refine_mask.unsqueeze(2)
    refine_ohkm_loss = torch.sum(refine_ohkm_loss)/(w*h*torch.sum(refine_mask))

    refine_ohem_loss,_ = torch.sort(loss_heatmap,dim=0,descending=True)
    refine_ohem_loss = torch.sum(refine_ohem_loss[:int(batch_size * 0.5)])/(w*h*torch.sum(refine_weights))

    total_loss = 1*global_ohkm_loss + 1*global_ohem_loss + 8*(refine_ohkm_loss + refine_ohem_loss)
    info = {
        'global_ohkm_loss': global_ohkm_loss.data[0],
        'global_ohem_loss': global_ohem_loss.data[0],
        'refine_ohkm_loss': refine_ohkm_loss.data[0],
        'refine_ohem_loss': refine_ohkm_loss.data[0],
        'total_loss': total_loss.data[0],
    }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, step)

    return total_loss



