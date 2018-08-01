import  tensorflow as tf
from  slim.nets import resnet_v2
import  tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from  utils.utils import  *
from  model.prm import pyr_residual


class RefineNet_Pyr():
    def __init__(self,global_net):
        print('Build refine network...')
        self.global_net = global_net
        self.with_ohem=global_net.with_ohem
        self.training = global_net.training
        self.out_size = global_net.out_size
        self.joints_num = global_net.joints_num
        self.weights = tf.placeholder(dtype=tf.float32, shape=(None, self.joints_num))  ## 关键点的权重,有的关键点是不可见的
        self.gt = tf.placeholder(dtype=tf.float32, shape=(None,4, self.out_size[0], self.out_size[1], self.joints_num))
        self.fea = self.get_refine_concat_fea_plus(global_net.pyramid)
        self.output = self.build_output(self.fea)
        self.ohem_k = int(self.joints_num*0.7)
        self.l2_loss = self.get_loss(self.ohem_k)
        print('Build refine network finished!')

    def get_loss(self,ohem_k):
        with tf.name_scope('refine_loss'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.gt)
            w1 = tf.expand_dims(self.weights, axis=1)
            w2 = tf.expand_dims(w1, axis=1)
            w3 = tf.expand_dims(w2, axis=1)
            if self.with_ohem:
                print('Refine plus net with ohem !')
                l2_loss = tf.reduce_mean(tf.multiply(w3, loss), axis=[0, 1, 2, 3])
                l2_ohem_loss, _ = tf.nn.top_k(l2_loss, self.ohem_k)
                # batch_size = loss.get_shape().as_list()[0]
                samples_ohem_n = 9
                samples_loss = tf.reduce_mean(tf.multiply(w3, loss), axis=[1, 2, 3, 4])
                samples_loss, _ = tf.nn.top_k(samples_loss, samples_ohem_n)
                ohem_loss = tf.add(tf.reduce_mean(l2_ohem_loss), tf.reduce_mean(samples_loss), name='ohem_loss')
                ##l2_ohem_loss = tf.reduce_mean(l2_ohem_loss, name='ohem_loss')
                return ohem_loss
            else:
                print('Refine plus net no ohem !')
                l2_loss = tf.reduce_mean(tf.multiply(w3, loss, name='mseW'), name='L2_loss')
                return l2_loss

    def build_output(self, fea):
        arg_scope = extra_conv_arg_scope()
        with tf.variable_scope('refine_output'):
            with slim.arg_scope(arg_scope):
                output = [None] * 4
                for i, name in enumerate(self.refine_pyramid):
                    print(name)
                    p = slim.conv2d(self.refine_pyramid['c'], self.joints_num, [1, 1], stride=1, scope='p%d' % i,
                                    activation_fn=None)
                    p = tf.image.resize_bilinear(p, [self.out_size[0], self.out_size[1]], name='output%d' % i)
                    output[i] = p
                return tf.stack(output, axis=1, name='output')
    # def build_output(self,fea):
    #     arg_scope = extra_conv_arg_scope()
    #     with tf.variable_scope('refine_output'):
    #         out = [None] * 4
    #         with slim.arg_scope(arg_scope):
    #            out[0] = slim.conv2d(fea, self.joints_num, [1, 1], stride=1,activation_fn=None)
    #            out[1] = slim.conv2d(self.r1, self.joints_num, [1, 1], stride=1,activation_fn=None)
    #            out[2] = slim.conv2d(self.r2, self.joints_num, [1, 1], stride=1, activation_fn=None)
    #            out[3] = slim.conv2d(self.r3, self.joints_num, [1, 1], stride=1, activation_fn=None)
    #            return tf.stack(out, axis=1, name='refine_output')

    def get_refine_concat_fea(self,pyramid):
        f1 = pyramid['P2']
        f2 = pyramid['P3']
        f3 = pyramid['P4']
        f4 = pyramid['P5']

        with tf.variable_scope('refine_fea'):
                inputRes = (f1.get_shape().as_list()[1], f1.get_shape().as_list()[2])
                self.r1 = pyr_residual(f1, 128, 'no_preact', inputRes, 4, 6, 'r1', self.training)

                inputRes = (f2.get_shape().as_list()[1], f2.get_shape().as_list()[2])
                self.r2 = pyr_residual(f2, 128, 'no_preact', inputRes, 4, 6, 'r2', self.training)
                size =self.r2.get_shape().as_list()
                self.r2 = tf.image.resize_bilinear(self.r2, [int(2 * size[1]), int(2* size[2])])

                inputRes = (f3.get_shape().as_list()[1], f3.get_shape().as_list()[2])
                self.r3 = pyr_residual(f3, 128, 'no_preact', inputRes, 4, 4, 'r3', self.training)
                size = self.r3.get_shape().as_list()
                self.r3 = tf.image.resize_bilinear(self.r3, [int(4 * size[1]), int(4 * size[2])])

                inputRes = (f4.get_shape().as_list()[1], f4.get_shape().as_list()[2])
                self.r4 = pyr_residual(f4, 128, 'no_preact', inputRes, 4, 4, 'r4', self.training)
                size = self.r4.get_shape().as_list()
                self.r4= tf.image.resize_bilinear(self.r4, [int(8 * size[1]), int(8 * size[2])])
                concat = tf.concat([self.r1, self.r2, self.r3, self.r4], axis=3)
                fea = resnet_v2.bottleneck(concat, 256, 256, stride=1)

                return fea

    def get_refine_concat_fea_plus(self, pyramid):
        f1 = pyramid['P2']
        f2 = pyramid['P3']
        f3 = pyramid['P4']
        f4 = pyramid['P5']

        with tf.variable_scope('refine_fea'):
            with slim.arg_scope([slim.conv2d],
                    weights_regularizer=slim.l2_regularizer(0.00001)) as scope:
                inputRes = (f4.get_shape().as_list()[1], f4.get_shape().as_list()[2])
                self.r4 = pyr_residual(f4, 512, 'no_preact', inputRes, 8, 2, 'r4', self.training)
                size = self.r4.get_shape().as_list()
                self.r4 = tf.image.resize_bilinear(self.r4, [int(2 * size[1]), int(2 * size[2])])
                f3 = tf.add(f3, self.r4)
                inputRes = (f3.get_shape().as_list()[1], f3.get_shape().as_list()[2])
                self.r3 = pyr_residual(f3, 512, 'no_preact', inputRes, 8, 2, 'r3', self.training)
                size = self.r3.get_shape().as_list()
                self.r3 = tf.image.resize_bilinear(self.r3, [int(2 * size[1]), int(2 * size[2])])

                f2 = tf.add(f2, self.r3)
                inputRes = (f2.get_shape().as_list()[1], f2.get_shape().as_list()[2])
                self.r2 = pyr_residual(f2, 256, 'no_preact', inputRes, 8, 2, 'r2', self.training)
                size = self.r2.get_shape().as_list()
                self.r2 = tf.image.resize_bilinear(self.r2, [int(2 * size[1]), int(2 * size[2])])

                inputRes = (f1.get_shape().as_list()[1], f1.get_shape().as_list()[2])
                self.r1 = pyr_residual(f1, 256, 'no_preact', inputRes, 4, 4, 'r1', self.training)

                concat = tf.concat([self.r1, self.r2], axis=3)
                fea = slim.conv2d(concat, 512, [3, 3], stride=1)

                self.refine_pyramid = {}
                self.refine_pyramid['r1'] = self.r1
                self.refine_pyramid['r2'] = self.r2
                self.refine_pyramid['r3'] = self.r3
                self.refine_pyramid['c'] = fea
                return fea


