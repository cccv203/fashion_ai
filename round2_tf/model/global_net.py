import  tensorflow as tf
from  slim.nets import resnet_v2
import  tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from slim.nets import  inception_resnet_v2

from  utils.utils import  *
basenet_map = {
  'resnet50': {'C1':'resnet_v2_50/conv1',
                'C2':'resnet_v2_50/block1/unit_2/bottleneck_v2',
                'C3':'resnet_v2_50/block2/unit_3/bottleneck_v2',
                'C4':'resnet_v2_50/block3/unit_5/bottleneck_v2',
                'C5':'resnet_v2_50/block4/unit_3/bottleneck_v2',
               }
}


class GlobalNet():
    def __init__(self,input_size=(256,256),joints_num=12,training=True,with_ohem=False):
        self.out_size= (int(input_size[0]/4),int(input_size[1]/4) )
        self.joints_num = joints_num
        self.training = training
        self.with_ohem=with_ohem
        self.ohem_k = int(self.joints_num * 0.7)
        self.base_fea_n = 512
        self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, input_size[0], input_size[1], 3), name='inputs')

        self.weights = tf.placeholder(dtype=tf.float32, shape=(None, self.joints_num))  ## 关键点的权重,有的关键点是不可见的
        self.out_N = 4  ##  C2~C5
        self.gt = tf.placeholder(dtype=tf.float32, shape=(None, self.out_N, self.out_size[0], self.out_size[1], self.joints_num))

        print('Now build global net graph...')
        self.output = self.build_graph()
        self.l2_loss = self.get_loss()
        #self.avg_output = tf.reduce_mean(self.output,axis=1)
        self.avg_output = self.output[:,3,:,:,:]
        reg_loss = tf.add_n(slim.losses.get_regularization_losses())
        self.loss = tf.add(reg_loss,self.l2_loss, name='total_loss')


        print('Build global net graph finished!')

    def get_loss(self):
        with tf.name_scope('global_net_loss'):
            ### TO DO   modify to L2  loss

            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.gt)
            w1 = tf.expand_dims(self.weights, axis=1)
            w2 = tf.expand_dims(w1, axis=1)
            w3 = tf.expand_dims(w2, axis=1)
            if self.with_ohem:
                print('Global net with ohem !')
                l2_loss = tf.reduce_mean(tf.multiply(w3, loss), axis=[0, 1, 2, 3])
                l2_ohem_loss, _ = tf.nn.top_k(l2_loss, self.ohem_k)
                #batch_size = loss.get_shape().as_list()[0]
                samples_ohem_n = 10
                samples_loss = tf.reduce_mean(tf.multiply(w3, loss), axis=[1, 2, 3, 4])
                samples_loss,_= tf.nn.top_k(samples_loss, samples_ohem_n)
                ohem_loss = tf.add(tf.reduce_mean(l2_ohem_loss), tf.reduce_mean(samples_loss),name='ohem_loss' )
                ##l2_ohem_loss = tf.reduce_mean(l2_ohem_loss, name='ohem_loss')
                return ohem_loss
            else:
                print('Global net no ohem !')
                l2_loss = tf.reduce_mean(tf.multiply(w3, loss, name='mseW'), name='L2_loss')
                return l2_loss

            # l2_loss = tf.reduce_mean(tf.multiply(w3, loss, name='mseW'), name='L2_loss')
            # return l2_loss
            # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.gt),
            #                       name='cross_entropy_loss')
            # w1 = tf.expand_dims(self.weights, axis=1)
            # w2 = tf.expand_dims(w1, axis=1)
            # w3 = tf.expand_dims(w2, axis=1)
            # reg_loss = tf.reduce_mean(slim.losses.get_regularization_losses())
            # l2_loss = tf.reduce_mean(tf.multiply(w3, loss, name='lossW'), name='L2_loss')
            # return tf.add(l2_loss, reg_loss, name='total_loss')

    def get_res_network(self, inputs,name='resnet50', weight_decay=0.00001):
        #with inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=weight_decay):
        # _, end_points2 = inception_resnet_v2.inception_resnet_v2(inputs, is_training=self.training)
        # for i in end_points2:
        #     print(end_points2[i])

        if name == 'resnet50':
            with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
                logits, end_points = resnet_v2.resnet_v2_50(inputs, is_training=self.training)


        else:
            print('now only support resnet 50')
            ###  TO  DO
            with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
                logits, end_points = resnet_v2.resnet_v2_50(inputs,  is_training=self.training)

        return  logits,end_points

    def build_pyramid(self,end_points,net_name='resnet50'):
        pyramid = {}
        pyramid_map = basenet_map[net_name]
        arg_scope = extra_conv_arg_scope()
        with tf.variable_scope('pyramid'):
            with slim.arg_scope(arg_scope):
                pyramid['P5'] =slim.conv2d(end_points[pyramid_map['C5']], self.base_fea_n, [1, 1], stride=1, scope='C5')
                for i in range(4, 1, -1):
                    #print( end_points[pyramid_map['C%d' % (i)]] )
                    p, c= pyramid['P%d' % (i + 1)], end_points[pyramid_map['C%d' % (i)]]
                    up_shape = tf.shape(c)

                    ###  where CPN  different from fpn
                    #  Slightly different from FPN, we apply 1 × 1 convolutional kernel
                    #  before each element-wise sum procedure in the upsampling process
                    p = slim.conv2d(p, self.base_fea_n, [1, 1], stride=1, scope='P%d' % i)

                    p = tf.image.resize_bilinear(p, [up_shape[1], up_shape[2]], name='C%d/upscale' % i)
                    c = slim.conv2d(c, self.base_fea_n, [1, 1], stride=1, scope='C%d' % i)
                    p = tf.add(p, c, name='C%d/addition' % i)
                    p = slim.conv2d(p, self.base_fea_n, [3, 3], stride=1, scope='C%d/fusion' % i)
                    pyramid['P%d' % (i)] = p
                return pyramid
    def build_output(self,pyramid):
        ## output0  is the top output
        ##
        arg_scope = extra_conv_arg_scope()
        with tf.variable_scope('output'):
            with slim.arg_scope(arg_scope):
                output= [None] *  self.out_N
                for i,name in enumerate(pyramid):
                    p = slim.conv2d(pyramid[name], self.joints_num, [1, 1], stride=1, scope='p%d' % i,activation_fn=None)
                    p = tf.image.resize_bilinear(p, [ self.out_size[0], self.out_size[1] ],name= 'output%d' % i)
                    output[i] = p
                return tf.stack(output, axis=1, name='final_output')



    def build_graph(self):

        _,end_points = self.get_res_network(self.inputs)
        self.pyramid = self.build_pyramid(end_points)
        output = self.build_output(self.pyramid)

        return output

    


if __name__=='__main__':
    net = GlobalNet(input_size=(384,188))
