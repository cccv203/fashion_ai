import numpy as np
from code.dataset.data_common import *
import imgaug as ia
from imgaug import augmenters as iaa
import random
from torch.utils.data import Dataset
import torch
from  code.config import  cfg
from  code.utils.vis import  *

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
class DataProvider(Dataset):
    def __init__(self,cfg=cfg):
        self.cfg = cfg
        self.except_list = []

        self.data_dict, self.joints_num = get_train_data_dict(cfg.joints_file, cfg.group, cfg.which,
                                                              except_list=self.except_list, flip=False)
        self.bbox_dict = build_bbox_dict(cfg.bbox_file)
        #  在 data dict 中，并且有框的标注

        self.train_img_list = list(
            self.data_dict.keys())  # [i for i in self.data_dict.keys() if i in self.bbox_dict.keys() and i not in self.err_list]



        for name in self.data_dict.keys():
            if (not (np.array_equal(self.data_dict[name]['joints'][7], [2000, 2000]))) or (
            not (np.array_equal(self.data_dict[name]['joints'][19], [2000, 2000]))):
                self.train_img_list.append(name)
                self.train_img_list.append(name)
        '''
        add some hard sample three times for train
        '''
        # self.err_list = get_err_list(self.cfg.err_file)
        # for err_name in self.err_list:
        #     self.train_img_list.append(err_name)
        #     self.train_img_list.append(err_name)
        random.shuffle(self.train_img_list)
        self.img_weight = {}
        for n in self.train_img_list:
            self.img_weight[n] = 1.0

        self.train_global_inx = 0
        self.joints_w = [1] * self.joints_num
        # self.joints_w[5]=2.0
        # self.joints_w[6] = 2.0
        # self.joints_w[7] = 2.0
        # self.joints_w[8] = 2.0
        # self.joints_w[14] = 2.0
        # self.joints_w[19] = 2.0
        print('total samples: ', len(self.train_img_list))

    def data_agument(self, image, joints, bb_joints=None):
        ia.seed(1)
        pts = []
        for j in joints:
            pts.append(ia.Keypoint(x=j[0], y=j[1]))
        keypoints = ia.KeypointsOnImage(pts, shape=image.shape)
        r = self.cfg.train_img_size[0] / self.cfg.train_img_size[1]
        pad = max(image.shape[0], image.shape[1]) * min(r, 1 / r) - min(image.shape[0], image.shape[1])
        pad = int(pad / 2)
        if pad < 0:
            pad = 0
        if image.shape[0] > image.shape[1]:
            px = (0, pad, 0, pad)
        else:
            px = (pad, 0, pad, 0)
        angle = np.random.randint(-25, 25)
        rr = image.shape[0]/image.shape[1]
        if rr>1.5 or rr<1/1.5:
            angle = np.random.randint(-32, 32)
        if rr>1.75 or rr<1/1.75:
            angle = np.random.randint(-16, 16)
        if rr>2.0 or rr<1/2.0:
            angle = np.random.randint(-8, 8)
        scale = 1# + 0.001 * np.random.randint(-100, 100)
        seq = iaa.Sequential([
            iaa.CropAndPad(px=px, pad_cval=114),
            iaa.Affine(rotate=angle,scale=scale, cval=114),
            iaa.Scale({'height': self.cfg.train_img_size[0], 'width': self.cfg.train_img_size[1]},
                      )
        ])
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([image])[0]
        keypoints_aug = seq_det.augment_keypoints([keypoints])[0]
        aug_joints = np.zeros(shape=joints.shape, dtype=np.int32)
        for j in range(aug_joints.shape[0]):
            aug_joints[j][0] = int(keypoints_aug.keypoints[j].x)
            aug_joints[j][1] = int(keypoints_aug.keypoints[j].y)
        if bb_joints is None:
            return image_aug, aug_joints
        else:
            pts2 = []
            for j in bb_joints:
                pts2.append(ia.Keypoint(x=j[0], y=j[1]))
            keypoints2 = ia.KeypointsOnImage(pts2, shape=image.shape)
            keypoints_aug2 = seq_det.augment_keypoints([keypoints2])[0]
            aug_joints2 = np.zeros(shape=bb_joints.shape, dtype=np.int32)
            for j in range(aug_joints2.shape[0]):
                aug_joints2[j][0] = int(keypoints_aug2.keypoints[j].x)
                aug_joints2[j][1] = int(keypoints_aug2.keypoints[j].y)
            return image_aug, aug_joints, aug_joints2

    def get_gaussian(self, height, width, sigma=1, center=None):
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2/2)

    def get_expand_roi_img(self,name):
        if name[-5:] == '_flip':
            img = cv2.imread(self.cfg.imgdir + name[:-5])
            img = cv2.flip(img, 1)
            joints = self.data_dict[name]['joints'].copy()
            joints[:, 0] = img.shape[1] - joints[:, 0]
            b = self.bbox_dict[name[:-5]].copy()
            b[0]= img.shape[1] -b[0]
            b[2] = img.shape[1] - b[2]
            t = b[0]
            b[0]=b[2]
            b[2] = t
        else:
            img = cv2.imread(self.cfg.imgdir + name)
            joints = self.data_dict[name]['joints'].copy()
            b = self.bbox_dict[name].copy()

        w = b[2]-b[0]
        h = b[3]-b[1]
        assert  w>0
        assert  h>0
        extUp = h * 0.01 * np.random.randint(self.cfg.extUp[0], self.cfg.extUp[1])
        extDown = h * 0.01 * np.random.randint(self.cfg.extDown[0],self.cfg.extDown[1])
        extL = w * 0.01 * np.random.randint(self.cfg.extL[0],self.cfg.extL[1])
        extR = w * 0.01 * np.random.randint(self.cfg.extR[0],self.cfg.extR[1])
        b[0] = b[0] - extL
        b[1] = b[1] - extUp
        b[2] = b[2] + extR
        b[3] = b[3] + extDown
        b[0] = b[0] if b[0] > 0 else 0
        b[1] = b[1] if b[1] > 0 else 0
        b[2] = b[2] if b[2] < img.shape[1] else img.shape[1]
        b[3] = b[3] if b[3] < img.shape[0] else img.shape[0]
        joints_after = np.round(joints - [b[0], b[1]])
        img = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
        return img, list(map(int, b)), np.array(joints_after, dtype=np.int32)

    def get_one_img(self, name):
        img, bb, joints = self.get_expand_roi_img(name)
        img, joints = self.data_agument(img, joints)
        return img, joints

    def calculate_iou(self,gt_point,rec_filed,box2):
        gt_point_ul_x = 0 if gt_point[0]-rec_filed/2 <0 else gt_point[0]-rec_filed/2
        gt_point_ul_y = 0 if gt_point[1]-rec_filed/2 <0 else gt_point[1]-rec_filed/2
        gt_point_br_x = self.cfg.hm_size[1] if gt_point[0] + rec_filed / 2 > self.cfg.hm_size[1] else gt_point[0] + rec_filed / 2
        gt_point_br_y = self.cfg.hm_size[0] if gt_point[1] + rec_filed / 2 > self.cfg.hm_size[0] else gt_point[1] + rec_filed / 2

        width = gt_point_br_x - gt_point_ul_x +box2[2]-box2[0] - (max(gt_point_br_x,box2[2]) - min(gt_point_ul_x,box2[0]))
        height = gt_point_br_y - gt_point_ul_y +box2[3]-box2[1] - (max(gt_point_br_y,box2[3]) - min(gt_point_ul_y,box2[1]))
        if width <= 0 or height <= 0:
            ratio = 0
        else:
            Area = width * height
            Area1 = (gt_point_br_x - gt_point_ul_x) * (gt_point_br_y - gt_point_ul_y)
            Area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            ratio = Area * 1. / (Area1 + Area2 - Area)
        return ratio

    def get_receptive_filed(self,keypoint,rec_filed):
        ul_x = 0 if keypoint[0] - rec_filed / 2 < 0 else keypoint[0] - rec_filed / 2
        ul_y = 0 if keypoint[1] - rec_filed / 2 < 0 else keypoint[1] - rec_filed / 2
        br_x = self.cfg.hm_size[1] if keypoint[0] + rec_filed / 2 > self.cfg.hm_size[1] else keypoint[0] + rec_filed / 2
        br_y = self.cfg.hm_size[0] if keypoint[1] + rec_filed / 2 > self.cfg.hm_size[0] else keypoint[1] + rec_filed / 2
        return [ul_x,ul_y,br_x,br_y]


    def get_heatmap(self, joints, weights):
        r = 1
        kernel_list = [(3 * r,3 * r), (3 * r, 3 * r), (3 * r, 3 * r), (3 * r, 3 * r)]
        size = self.cfg.hm_size
        joints_num = joints.shape[0]
        hm = np.zeros((len(kernel_list), size[0], size[1], joints_num), dtype=np.float32)
        for i in range(joints_num):
            if not (np.array_equal(joints[i], [-1, -1])) and weights[i] > 0:
                center = (joints[i, 0] / self.cfg.img_hm_r[0], joints[i, 1] / self.cfg.img_hm_r[1])
                if center[0]>self.cfg.hm_size[1]-1 or \
                        center[1]>self.cfg.hm_size[0]-1 or\
                        center[0]<0 or center[1]<0:
                    weights[i] = 0.0 ## 相当于不存在？ 不存在的点要监督期全零输出 但是权重不能为1，尤其是mix时
                    continue

                for j in range(len(kernel_list)):
                    kernel = kernel_list[j]
                    mp = self.get_gaussian(self.cfg.hm_size[0], self.cfg.hm_size[1], 3, center)
                    norm = np.max(mp)
                    if norm > 0.1:
                        mp = mp / norm
                    else:
                        mp = 0
                    hm[j, :, :, i] = mp
        r_hm = np.zeros((size[0], size[1], joints_num), dtype=np.float32)
        if self.cfg.use_rec_filed:
            r_hm = hm[-1]
        else:
            for i in range(joints_num):
                if not (np.array_equal(joints[i], [-1, -1])) and weights[i] > 0:
                    center = (joints[i, 0] / self.cfg.img_hm_r[0], joints[i, 1] / self.cfg.img_hm_r[1])
                    mp = self.get_gaussian(self.cfg.hm_size[0], self.cfg.hm_size[1], 5, center)
                    norm = np.max(mp)
                    if norm > 0.1:
                        mp = mp / norm
                    else:
                        mp = 0
                    r_hm[:, :, i] = mp
        return hm, r_hm,weights

    def get_one_train_img_and_hm(self, name):
        img, joints = self.get_one_img(name)
        w = self.joints_w.copy()  ## global
        for i, v in enumerate(self.data_dict[name]['visible']):
            if v == -1:  ## 不存在， mix 时设置为0  非mix 可设置为0到1
                w[i] = 0.0  ## 不存在的点要监督期全零输出 但是权重不能为1，尤其是mix时
            if v == 0:  ## 遮挡的点
                w[i] = w[i] * 1.* self.img_weight[name]  # global  考虑少一点
            if v == 1:
                w[i] = w[i] * self.img_weight[name]
        w2 = self.joints_w.copy()  ## refine
        for i, v in enumerate(self.data_dict[name]['visible']):
            if v == -1:  ## 不存在， mix 时设置为0  非mix 可设置为0到1
                w2[i] = 0.0  ## 不存在的点要监督期全零输出 但是权重不能为1，尤其是mix时
            if v == 0:  ##
                w2[i] = w2[i] * 1. * self.img_weight[name]  # w2[i] * self.cfg.unvisible_pts_weight * self.img_weight[name]
            if v == 1:
                w2[i] = w2[i] * self.img_weight[name]
        hm, r_hm,w = self.get_heatmap(joints, w)
        w2=w.copy()
        return img, hm, r_hm, w, w2, joints

    def get_current_train_name_list(self, batch_size):
        if self.train_global_inx + batch_size <= len(self.train_img_list):
            names = self.train_img_list[self.train_global_inx:self.train_global_inx + batch_size]
            self.train_global_inx = self.train_global_inx + batch_size
            return names
        else:
            n = self.train_global_inx + batch_size
            r = n - len(self.train_img_list)
            names = self.train_img_list[0:r]
            for i in range(self.train_global_inx, len(self.train_img_list), 1):
                names.append(self.train_img_list[i])
            self.train_global_inx = r
            return names

    def __len__(self):
        return len(self.train_img_list)

    def __getitem__(self, item):
        name = self.train_img_list[item]
        img, hm, r_hm, w, w2, joints = self.get_one_train_img_and_hm(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.cfg.vis:
            visible = self.data_dict[name]['visible']
            hm_gui = np.sum(hm[0], axis=2)
            draw_hm(img, hm_gui)
            draw_joints(img, joints, visible=visible)
        img = img.astype(np.float32)
        img[:, :, 0] = img[:, :, 0] - _R_MEAN
        img[:, :, 1] = img[:, :, 1] - _G_MEAN
        img[:, :, 2] = img[:, :, 2] - _B_MEAN

        img = img.transpose(2,0,1)
        hm = hm.transpose(0,3,1,2)
        r_hm = r_hm.transpose(2,0,1)

        img, hm, r_hm = torch.from_numpy(img).float(),torch.from_numpy(hm).float(),torch.from_numpy(r_hm).float()
        w, w2, joints = torch.FloatTensor(w),torch.FloatTensor(w2),torch.FloatTensor(joints),
        return img,hm,r_hm,w,w2,joints


if __name__=='__main__':
    cfg.vis = True
    data = DataProvider(cfg)
    for i,(img,hm,r_hm,w,w2,joints) in enumerate(data):
        # print((hm))
        continue


