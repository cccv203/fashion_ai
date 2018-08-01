from  test_framework import  *
import tensorflow as tf
import tensorflow.contrib.slim  as slim
import  argparse
from  net.network_plus_plus import Network
from  test_framework import  TestData
import  numpy as np
import cv2
import sys
from data.data_common import  *
from net.generator_plus_plus import generator
parse = argparse.ArgumentParser()

parse.add_argument('--img_dir',type=str,default="/home/cccv/disk/key_point/fashionAI/train/")
parse.add_argument('--test_model_dir',type=str,default='./log')
parse.add_argument('--test_csv',type=str,default='./csv/fashionAI_key_points_test_b_answer_20180426.csv')
parse.add_argument('--save',type=str,default='pred_b.csv')
parse.add_argument('--bbox_file',type=str,default='./csv/detect_bbox.csv')
FLAGS = parse.parse_args()

img_h = 384
img_w = 288

# img_dir="F:\\round2_fashionAI_key_points_test_a_20180426\\"
# test_csv="F:\\round2_fashionAI_key_points_test_a_20180426\\test.csv"

class Net():
    def __init__(self,joints_num):
        self.inputs = tf.placeholder(dtype=tf.float32,shape=(1,img_h,img_w, 3), name='inputs')
        self.g_out, self.r_out, self.g_var_restore = generator(self.inputs,
                                                               training=False,
                                                               basenet='resnet101',
                                                               out_size=(int(img_h/4),int(img_w/4)),
                                                               joints_num=joints_num)

class TestManager():
    def __init__(self,net):
        print('Init test manager...')
        self.sess = tf.Session()
        self.net = net
        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.test_model_dir)#
        if ckpt and ckpt.model_checkpoint_path:
            print('restore from ', ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('No trained model in ', FLAGS.test_model_dir)
            exit(0)
        print('Init test manager finished!')

    def get_pred_joints(self,src,imgRGB):
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        img = imgRGB.astype(np.float32)
        img[:, :, 0] = img[:, :, 0] - _R_MEAN
        img[:, :, 1] = img[:, :, 1] - _G_MEAN
        img[:, :, 2] = img[:, :, 2] - _B_MEAN
        inputs = np.zeros((1, img.shape[0], img.shape[1], 3), dtype=np.float32)
        inputs[0] = img
        out=self.net.r_out #+self.net.g_out[:,3,:,:,:]
        #out = tf.image.resize_bilinear(out,(img_h,img_w))
        output = self.sess.run(out,   feed_dict={self.net.inputs: inputs })
        output = output[0]
        # show = np.sum(output,axis=2)
        # cv2.namedWindow('show',0)
        # cv2.imshow('show',show)
        # cv2.waitKey()
        preds = []
        for i in range(output.shape[2]):
            o = output[:,:,i]
            #p = np.unravel_index(o.argmax(), o.shape)
            #print(np.max(o))
            # o=cv2.resize(o,(img.shape[1],img.shape[0]))
            pp=[]
            for k in range(10):
                p0 = np.unravel_index(o.argmax(), o.shape)
                o[p0] = 0.0
                pp.append(p0)
            p=[0,0]
            for item in pp:
                p[0]+=item[0]
                p[1]+=item[1]
            p[0]/=len(pp)
            p[1]/=len(pp)
            #preds.append([p[0] , p[1] ])
            preds.append([p[0]*img.shape[0]/o.shape[0], p[1]*img.shape[1]/o.shape[1] ])
        return  preds



def  test():
    catgory_list=data_map['mix']['mix5']['catgory_list']
    inx_list = data_map['mix']['mix5']['inx_list']
    joints_num = len(inx_list)
    data = TestData(FLAGS.img_dir, FLAGS.test_csv, catgory_list,inx_list,
                    img_size=(img_h,img_w),bbox_file=FLAGS.bbox_file,expandS=0.2)
    data_iter = data.generator(color_mode='RGB')
    ##-----------------------------------------------------
    ## 以下是初始化模型  不同框架不一样
    net = Net(joints_num=joints_num)
    model = TestManager(net)
    ##-----------------------------------------------------

    # 遍历data 集合，利用模型检测预处理过后的图 preprocessed
    # src  原图
    # preprocessed   预处理后的图  模型预测只需要关注这个图
    i=0
    for src, preprocessed,name in data_iter:
        i=i+1
        sys.stdout.write('\rImage  ' + str(i) + '   ' + name)
        sys.stdout.flush()
        #print('Image  '+str(i)+'   ', name)
        ##  这里不同框架的形式不一样
        ##  我这的输出是  [[y,x], [y,x] .....]  y代表在行上的偏移
        #print(preprocessed.shape)
        pred = model.get_pred_joints(src,preprocessed)
        ## ------------------------------------------
        ##  每次检测过后一定要登记检测结果
        data.reg_pred(src,name,pred,isXFisrt=False)

        data.draw_final_pred(src,name)
        # cv2.imwrite('0013.png',src)
        cv2.namedWindow('src', 0)
        cv2.imshow('src', src)
        cv2.waitKey(0)
    data.write_result_to_csv(FLAGS.save)
    print('finished!')


if __name__=='__main__':
    test()
