import  tensorflow as tf
import  tensorflow.contrib.slim as slim

from net.generator_plus_plus import generator
from net.loss import *
import  time
import numpy as np
import sys
import os
from  config import  cfg


class Network():
    def __init__(self,cfg=cfg):
        print('Build network.....')
        self.cfg=cfg
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=(self.cfg.batch_size,
                                            self.cfg.train_img_size[0],
                                            self.cfg.train_img_size[1], 3), name='inputs')

        self.global_weights = tf.placeholder(dtype=tf.float32,shape=(self.cfg.batch_size,self.cfg.joints_num))
        self.refine_weights = tf.placeholder(dtype=tf.float32, shape=(self.cfg.batch_size, self.cfg.joints_num))

        self.g_gt = tf.placeholder(dtype=tf.float32, shape=(self.cfg.batch_size, 4, self.cfg.hm_size[0],
                                                            self.cfg.hm_size[1], self.cfg.joints_num))
        self.r_gt = tf.placeholder(dtype=tf.float32, shape=(self.cfg.batch_size, self.cfg.hm_size[0],
                                                            self.cfg.hm_size[1], self.cfg.joints_num))

        self.g_out,self.r_out,self.g_var_restore = generator(self.inputs,
                                                             training=self.cfg.training,
                                                             basenet=self.cfg.basenet,
                                                             out_size=self.cfg.hm_size,
                                                             joints_num=self.cfg.joints_num)

        self.variables_mapping = {}
        variables = tf.model_variables()
        for variable in variables:
            key = variable.name.replace('generator/', '').replace('discriminator/', '')[:-2]
            if  'pyramid' not in key and 'g_out' not in key and 'r_out' not in key and 'resnet_v2_50/logits' not in key:
                self.variables_mapping[key] = variable


        self.heatmap_loss = heatmap_loss(g_out=self.g_out,g_gt=self.g_gt,
                                     r_out=self.r_out,r_gt=self.r_gt,
                                     global_weights=self.global_weights,
                                     refine_weights=self.refine_weights,
                                     global_ohkm_n=int(0.8*self.cfg.joints_num),
                                    refine_ohkm_n=int(0.3 * self.cfg.joints_num),
                                     batch_size=self.cfg.batch_size)
        self.reg_loss = tf.add_n(tf.losses.get_regularization_losses())
        tf.summary.scalar('reg_loss',self.reg_loss)
        self.total_loss = self.reg_loss+ self.heatmap_loss
        tf.summary.scalar('total_loss', self.total_loss)
        self.train_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.cfg.base_lr, self.train_step,
                                             self.cfg.decay_steps, self.cfg.decay_factor,
                                             staircase=False, name='learning_rate')
        self.optmizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optmizer.minimize(self.total_loss, self.train_step)
        print('Build network down!')



    def load_pretrained_param(self):
        with tf.device('/cpu:0'):
            restorer = tf.train.Saver(self.variables_mapping)
            restorer.restore(self.sess, self.cfg.finetune_model)

    def train(self,data_iter):
        with tf.name_scope('train'):
                self.sess = tf.Session()
                self.saver = tf.train.Saver()
                self.train_writer = tf.summary.FileWriter(self.cfg.train_dir, self.sess.graph)
                init_op = tf.global_variables_initializer()
                self.sess.run(init_op)
                ckpt = tf.train.get_checkpoint_state(self.cfg.train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print('restore from ', ckpt.model_checkpoint_path)
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                    start_epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('_')[-2])
                    global_step_np = (start_epoch)*self.cfg.epoch_size
                    self.sess.run(tf.assign(self.train_step, global_step_np))
                else:
                    start_epoch =0
                    print('No past checkpoint file found, using pretrained model in %s ' % self.cfg.finetune_model)
                    self.load_pretrained_param()
                    print('Load pretrained model finished!')
                merged = tf.summary.merge_all()

                for epoch in range(self.cfg.epoches):
                    epochStartTime = time.time()
                    lr=self.cfg.base_lr
                    at_step=0
                    print('\r Epoch :' + str(epoch) + '/' + str(self.cfg.epoches))
                    for i in range(self.cfg.epoch_size):
                        percent = ((i + 1) / self.cfg.epoch_size) * 100
                        num = np.int(10 * percent / 100)
                        tToEpochEnd = int((time.time() - epochStartTime) * (100 - percent) / (percent))
                        sys.stdout.write(
                            '\r Progress: {0}>'.format(">" * num) + "{0}>".format(" " * (10 - num)) + '||' +
                            str(percent)[:3] + '%' +
                            '\t | wait: ' + str(tToEpochEnd) + ' sec.'+
                             '\t | lr: '+str(lr) +' | step: '+str(at_step) )
                        sys.stdout.flush()
                        train_imgs, global_gtmap, refine_gtmap, global_weights, refine_weights  = next(data_iter)
                        _,loss,summary, lr, at_step = \
                                self.sess.run([self.train_op,self.total_loss, merged,
                                               self.lr, self.train_step],
                                              feed_dict={self.inputs: train_imgs,
                                                         self.g_gt:global_gtmap,
                                                         self.r_gt:refine_gtmap,
                                                         self.global_weights: global_weights,
                                                         self.refine_weights:refine_weights
                                                         })
                        if i%10==1 and i>100:
                            self.train_writer.add_summary(summary, at_step)
                    print('\nEpoch ' + str(epoch) + '/' + str(self.cfg.epoches) + ' done in ' + str( int(time.time() - epochStartTime)) + ' sec.')
                    checkpoint_path = os.path.join(self.cfg.train_dir,self.cfg.which+ '_fine_model_'+str(start_epoch+epoch+1)+'_epoch.ckpt')
                    save_path = self.saver.save(self.sess, checkpoint_path)
                    print("Save model in: %s" % save_path)




