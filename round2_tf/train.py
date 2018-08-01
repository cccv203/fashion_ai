import argparse
from config import  cfg
from net.network_plus_plus import Network
from data.dataflow import DataProvider
parse = argparse.ArgumentParser()
parse.add_argument('--group',type=str,default='mix')
parse.add_argument('--which',type=str,default='mix5')
parse.add_argument('--finetune_model',type=str,default='/home/cccv/disk/key_point/code_release/train/data/pretrained/resnet_v2_101.ckpt')
parse.add_argument('--train_csv',type=str,default='./csv/stage_1_train_and_test_a.csv')
parse.add_argument('--bbox_file',type=str,default='./csv/coarse_bbox.csv')
parse.add_argument('--img_dir',type=str,default="/home/cccv/disk/key_point/fashionAI/train")
parse.add_argument('--train_dir',type=str,default='./log')

parse.add_argument('--batch_size',type=int,default=10)
parse.add_argument('--crop_w',type=int,default=288)
parse.add_argument('--crop_h',type=int,default=384)
parse.add_argument('--base_lr',type=float,default=0.00005)
parse.add_argument('--decay_steps',type=int,default=40000)
parse.add_argument('--decay_factor',type=float,default=0.5)
parse.add_argument('--epoch_size',type=int,default=1000)
parse.add_argument('--epoches',type=int,default=80)
FLAGS = parse.parse_args()

def parse_config():
    cfg.batch_size = FLAGS.batch_size
    cfg.epoch_size = FLAGS.epoch_size
    cfg.train_dir = FLAGS.train_dir
    cfg.train_img_size = (FLAGS.crop_h,FLAGS.crop_w)
    cfg.hm_size = (int(FLAGS.crop_h/4),int(FLAGS.crop_w/4))
    cfg.group = FLAGS.group
    cfg.which = FLAGS.which
    cfg.base_lr = FLAGS.base_lr
    cfg.epoches = FLAGS.epoches
    cfg.joints_file= FLAGS.train_csv
    cfg.bbox_file=FLAGS.bbox_file
    cfg.finetune_model = FLAGS.finetune_model

if __name__=='__main__':
    parse_config()
    net = Network(cfg=cfg)
    df = DataProvider(cfg=cfg)
    dataIter = df.data_gen()
    net.train(dataIter)


