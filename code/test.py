import torch
from nets.cpn import CPN
from nets.cpn_v1 import CPN as CPN_v1
from nets.pyrnet import PyrNet
from code.config import cfg
from test_frame import TestData
from dataset.data_common import *
import numpy as np


def get_point(output):
    output = output.clone().cpu().data.numpy()
    preds = []
    for i in range(output.shape[0]):
        o = output[i]
        pp = []
        for k in range(2):
            p0 = np.unravel_index(o.argmax(), o.shape)
            o[p0] = 0.0
            pp.append(p0)
        p = [0, 0]
        for item in pp:
            p[0] += item[0]
            p[1] += item[1]
        p[0] /= len(pp)
        p[1] /= len(pp)
        preds.append([p[0] * 4+2, p[1] * 4+2])
    return  np.array(preds)

catgory_list=data_map['mix']['mix5']['catgory_list']
inx_list = data_map['mix']['mix5']['inx_list']
joints_num = len(inx_list)

if cfg.nets == 'cpn':
    model = CPN(cfg).cuda()
elif cfg.nets == 'cpn_v1':
    model = CPN_v1(cfg).cuda()
else:
    model = PyrNet(cfg).cuda()
# state_dict = torch.load('../logs/' + cfg.nets + '.pkl')
# model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})
# print('load model from '+'  ../logs/'+cfg.nets+'.pkl')
# model.eval()

testdata = TestData(cfg.imgdir, cfg.test_csv, catgory_list, inx_list, img_size=(512, 512), bbox_file='../data/csvs/detect_bbox.csv')
test_loader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False)
for i, (img,src,name) in enumerate(test_loader):
    inputs = torch.autograd.Variable(img.cuda())

    global_output, refine_output = model(inputs)
    pred_points = get_point(refine_output[0])
    testdata.reg_pred(np.array(src[0]), name[0], pred_points)
    print('save ' + str(i) + ' picture')
testdata.write_result_to_csv('pred_b.csv')


