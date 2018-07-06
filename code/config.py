from  code.dataset.data_common import *

class CFG():
    def __init__(self):

        self.training = True
        self.group='mix'
        self.which='mix5'
        self.catgory_list = data_map[self.group][self.which]['catgory_list']
        self.inx_list = data_map[self.group][self.which]['inx_list']
        self.joints_num = len(self.inx_list)

        self.nets = 'cpn_v11'

        self.cpn = {
            'basenet' : 'resnet101',
            'bottleneck' : 'resnet'
        }
        self.pyr = {
            'baseWidth' : 6,
            'cardinality' : 10,
            'nStack' : 5,
            'nFeats' : 256,
            'nResidual' : 1,

        }


        self.train_img_size = (512,512)
        self.hm_size=(128,128)
        self.img_hm_r=(4,4)

        ### 框随机裁取  按百分比
        self.extUp=(15,  25)
        self.extDown =(15,  25)
        self.extL = (15,  25)
        self.extR = (15,  25)


        self.batch_size=4
        self.epoch_size=1000

        self.base_lr = 5e-5
        self.decay_steps = 10000
        self.decay_factor =0.5
        self.weight_decay=1e-6

        self.epoches=80


        self.imgdir = "/home/cccv/disk/key_point/fashionAI/train/"
        self.joints_file="../data/csvs/stage_1_train_and_test_a.csv"
        self.bbox_file="../data/csvs/coarse_bbox.csv"
        self.test_csv = '../data/csvs/fashionAI_key_points_test_a_answer_20180426.csv'

        ##
        self.train_dir='./log/blouse_res101/'
        self.finetune_model=None

        self.err_file = '../data/csvs/train_hard.csv'

        self.vis=False
        self.basenet_pretrained = True
        self.base_fea_n = 256
        self.RESNET_FIXED_BLOCKS=0
        self.use_rec_filed = 107



cfg=CFG()
