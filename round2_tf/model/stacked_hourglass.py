import   tensorflow   as tf
import tensorflow.contrib.slim as slim
from  model.prm import  *

class Hourglass():
     def __init__(self, input_size=(256,256),out_size=(64,64),training=True,batch_size=12,feas = 512, stacks = 2, modules = 1, outputDim = 16):
          self.nLow =4
          self.input_size = input_size
          self.out_size = out_size
          self.training = training
          self.feas = feas
          self.stacks = stacks
          self.modules = modules
          self.outputDim = outputDim
          self.batch_size = batch_size
          self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, input_size[0], input_size[1], 3), name='inputs')
          self.weights_1 = tf.placeholder(dtype=tf.float32, shape=(None, self.outputDim))  ## stack  1 中关键点的权重
          self.weights_2 = tf.placeholder(dtype=tf.float32, shape=(None, self.outputDim)) ## stack 2 中关键点的权重
          self.gt = tf.placeholder(dtype=tf.float32, shape=(None, self.stacks, out_size[0], out_size[1], self.outputDim))

          ##  build graph
          self.output = self.build_graph()


     ## 两个stacked  hourglass 的loss
     def two_stacked_loss(self):
         o1 = self.output[:, 0, :, :, :]
         o2 = self.output[:, 1, :, :, :]
         g1 = self.gt[:, 0, :, :, :]
         g2 = self.gt[:, 1, :, :, :]
         self.loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o1, labels=g1),
                                     name='cross_entro_1')
         e1 = tf.expand_dims(self.weights_1, axis=1, name='expdim01')
         e1 = tf.expand_dims(e1, axis=1, name='expdim02')
         L1 = tf.multiply(e1, self.loss1, name='lossW1')
         self.loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o2, labels=g2),
                                     name='cross_entro_2')
         e2 = tf.expand_dims(self.weights_2, axis=1, name='expdim11')
         e2 = tf.expand_dims(e2, axis=1, name='expdim12')
         L2 = tf.multiply(e2, self.loss2, name='lossW2')
         w1 = tf.to_float(0.8)
         L1 = tf.multiply(L1, w1)
         self.files_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o2, labels=g2), axis=(1, 2, 3))
         return tf.reduce_mean(tf.add(L1, L2),name='loss')


     def c_b_r(self,inputs, numOut,ksize=1,stride=1,name='cbr'):
         with tf.name_scope(name):
             x = slim.conv2d(inputs, num_outputs=numOut, kernel_size=ksize,stride=stride, activation_fn=None)
             x = slim.batch_norm(inputs=x, decay=0.9, is_training=self.training, activation_fn=tf.nn.relu)
             return x

     def build_graph(self):
         print('Now build hourglass graph...')
         inputs = self.inputs
         with tf.name_scope('hourglass'):
                with tf.name_scope('pre'):
                     conv1 = slim.conv2d(inputs, 128, kernel_size=7, stride=2, activation_fn=None)
                     conv1 = slim.batch_norm(inputs=conv1, decay=0.9, is_training=self.training, activation_fn=tf.nn.relu)
                     inputsRes = self.input_size
                     res = (int(inputsRes[0] / 2), int(inputsRes[1] / 2))
                     r1 = pyr_residual(conv1, 128, 'no_preact', res, 6, 6, 'p1', self.training)
                     pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
                     res = (int(res[0] / 2), int(res[1] / 2))
                     r2 = pyr_residual(pool1, int(self.feas / 2), 'preact', res, 4, 6, 'p2', self.training)
                     r3 = pyr_residual(r2, self.feas, 'preact', res, 4, 6, 'p3', self.training)
                hg = [None] * self.stacks
                h1 = [None] * self.stacks
                h2 = [None] * self.stacks
                output = [None] * self.stacks
                sum_hidden = [None] * self.stacks
                with tf.name_scope('stacks'):
                     with tf.name_scope('stage_0'):
                          hg[0] = ori_hourglass(r3, self.nLow, self.feas, 'hourglass')
                          h1[0] = self.c_b_r(hg[0], self.feas, 1, 1, name='hl')
                          output[0]= slim.conv2d(h1[0],self.outputDim,kernel_size=1, stride=1, activation_fn=None)
                          h2[0] = slim.conv2d(output[0],self.feas,kernel_size=1, stride=1, activation_fn=None)
                          sum_hidden[0] = tf.add_n([h1[0], h2[0], r3], name='merge')
                     for i in range(1, self.stacks - 1):
                          with tf.name_scope('stage_' + str(i)):
                                hg[i] = prm_hourglass(sum_hidden[i - 1], self.nLow,
                                                      self.feas, res, 1, 'no_preact', 6, 6, 'hourglass', self.training)

                                h1[i] = self.c_b_r(hg[i], self.feas, 1, 1, name='hl')
                                output[i] = slim.conv2d(h1[i], self.outputDim, kernel_size=1, stride=1,
                                                        activation_fn=None)
                                h2[i] = slim.conv2d(output[i], self.feas, kernel_size=1, stride=1, activation_fn=None)
                                sum_hidden[i] = tf.add_n([h1[i], h2[i], sum_hidden[i-1]], name='merge')

                     with tf.name_scope('stage_' + str(self.stacks - 1)):
                          hg[self.stacks - 1] = prm_hourglass(sum_hidden[self.stacks - 2], self.nLow,
                                                              self.feas, res, 1, 'no_preact', 4, 6, 'hourglass', self.training)
                          h1[self.stacks - 1] = self.c_b_r(hg[self.stacks - 1], self.feas, 1, 1, name='ll')
                          output[self.stacks - 1] = slim.conv2d(h1[self.stacks - 1], self.outputDim, kernel_size=1, stride=1,
                                                  activation_fn=None)
                     print('Build hourglass graph finished!')

                     return tf.stack(output, axis=1, name='outputs')



if __name__ =='__main__':
    model = Hourglass()
    loss = model.two_stacked_loss()
    print(model.output)
    print(loss)



