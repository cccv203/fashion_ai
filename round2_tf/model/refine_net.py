import  tensorflow as tf
from  slim.nets import resnet_v2
import  tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from  utils.utils import  *

class RefineNet():
    def __init__(self,gt_size,pyramid,joints_num,training=True):
        print('Build refine network...')
        self.out_size = gt_size
        self.joints_num = joints_num
        self.weights = tf.placeholder(dtype=tf.float32, shape=(None, self.joints_num))  ## 关键点的权重,有的关键点是不可见的
        self.gt = tf.placeholder(dtype=tf.float32, shape=(None,4, self.out_size[0], self.out_size[1], self.joints_num))
        self.fea = self.get_refine_concat_fea(pyramid)
        self.output = self.build_output(self.fea)
        self.ohem_k = int(self.joints_num*0.7)
        self.l2_loss = self.get_loss(self.ohem_k)
        print('Build refine network finished!')

    def get_loss(self,ohem_k):
        with tf.name_scope('refine_loss'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.gt)
            print(loss)

            w1 = tf.expand_dims(self.weights, axis=1)
            w2 = tf.expand_dims(w1, axis=1)
            w3 = tf.expand_dims(w2, axis=1)

            l2_loss = tf.reduce_mean(tf.multiply(w3, loss),axis=[0,1,2,3])

            l2_ohem_loss,_ = tf.nn.top_k(l2_loss,ohem_k)
            print(l2_ohem_loss)

            l2_ohem_loss = tf.reduce_mean(l2_ohem_loss,name='ohem_loss')
            return l2_ohem_loss


    def build_output(self,fea):
        arg_scope = extra_conv_arg_scope()
        with tf.variable_scope('refine_output'):
            out = [None] * 4
            with slim.arg_scope(arg_scope):
               out[0] = slim.conv2d(fea, self.joints_num, [1, 1], stride=1,activation_fn=None)
               out[1] = slim.conv2d(self.f2, self.joints_num, [1, 1], stride=1,activation_fn=None)
               out[2] = slim.conv2d(self.f3, self.joints_num, [1, 1], stride=1, activation_fn=None)
               out[3] = slim.conv2d(self.f4, self.joints_num, [1, 1], stride=1, activation_fn=None)
               return tf.stack(out, axis=1, name='refine_output')

    def get_refine_concat_fea(self,pyramid):
        f1 = pyramid['P2']
        f2 = pyramid['P3']
        f3 = pyramid['P4']
        f4 = pyramid['P5']

        depth_bottleneck = 256
        arg_scope = extra_conv_arg_scope()
        with tf.variable_scope('refine_fea'):
            with slim.arg_scope(arg_scope):
                #f1 = resnet_v2.bottleneck(f1, 2 * depth_bottleneck, depth_bottleneck, stride=1)

                f2 = resnet_v2.bottleneck(f2, 2 * depth_bottleneck, depth_bottleneck, stride=1)

                size = f2.get_shape().as_list()
                f2 = tf.image.resize_bilinear(f2, [int(2 * size[1]), int(2 * size[2])])
                self.f2 = f2

                f3 = resnet_v2.bottleneck(f3, 2 * depth_bottleneck, depth_bottleneck, stride=1)
                f3 = resnet_v2.bottleneck(f3, 2 * depth_bottleneck, depth_bottleneck, stride=1)
                size = f3.get_shape().as_list()
                f3 = tf.image.resize_bilinear(f3, [int(4 * size[1]), int(4 * size[2])])
                self.f3 = f3

                f4 = resnet_v2.bottleneck(f4,  2 * depth_bottleneck, depth_bottleneck, stride=1)
                f4 = resnet_v2.bottleneck(f4,  2 * depth_bottleneck, depth_bottleneck, stride=1)
                f4 = resnet_v2.bottleneck(f4,  2 * depth_bottleneck, depth_bottleneck, stride=1)
                size = f4.get_shape().as_list()
                f4 = tf.image.resize_bilinear(f4, [int(8 * size[1]), int(8 * size[2])])
                self.f4 = f4
                concat = tf.concat([f1, f2, f3, f4], axis=3)
                fea = resnet_v2.bottleneck(concat, 2*depth_bottleneck, depth_bottleneck, stride=1)
                return fea

