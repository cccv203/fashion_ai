import  tensorflow as tf
from  slim.nets import resnet_v2
import  tensorflow.contrib.slim as slim
def extra_conv_arg_scope(weight_decay=0.00001, activation_fn=None, normalizer_fn=None):
    with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
            padding='SAME',
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn, ) as arg_sc:
        with slim.arg_scope(
                [slim.fully_connected],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn) as arg_sc:
            return arg_sc

def extra_conv_arg_scope_with_bn(weight_decay=0.00001,
                     activation_fn=None,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }
  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc