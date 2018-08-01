from  data.data_common import *

class CFG():
    def __init__(self):

        self.training = True
        self.group='mix'
        self.which='mix5'
        self.catgory_list = data_map[self.group][self.which]['catgory_list']
        self.inx_list = data_map[self.group][self.which]['inx_list']
        self.joints_num = len(self.inx_list)

        self.basenet='resnet101'
        self.train_img_size = (384,288)
        self.hm_size=(96,72)
        self.img_hm_r=(4,4)

        ### 框随机裁取  按百分比
        self.extUp=(15,  25)
        self.extDown =(15,  25)
        self.extL = (15,  25)
        self.extR = (15,  25)


        self.batch_size=24
        self.epoch_size=1000

        self.base_lr = 5e-4
        self.decay_steps =20000
        self.decay_factor =0.5

        self.epoches=80


        self.imgdir = "/home/cccv/disk/key_point/fashionAI/train/"
        self.joints_file="./csv/test.csv"
        self.bbox_file="./csv/coarse_bbox.csv"

        ##
        self.train_dir='./log/blouse_res101/'
        self.finetune_model=None

        self.err_file = './csv/train_hard.csv'
        self.use_rec_filed = 0


cfg=CFG()
