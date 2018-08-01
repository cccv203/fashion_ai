import sys
sys.path.append('./slim/')
import tensorflow as tf
from  slim.nets import resnet_v2
import tensorflow.contrib.slim as slim
from slim.nets import inception_resnet_v2


def extra_conv_arg_scope(weight_decay=1e-5, activation_fn=None, normalizer_fn=None):
    with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
            padding='SAME',
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=tf.truncated_normal_initializer(stddev=0.0001),
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn, ) as arg_sc:
        with slim.arg_scope(
                [slim.fully_connected],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=tf.truncated_normal_initializer(stddev=0.0001),
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn) as arg_sc:
            return arg_sc

basenet_map = {
    'resnet50': {'C1': 'generator/resnet_v2_50/conv1',
                 'C2': 'generator/resnet_v2_50/block1/unit_2/bottleneck_v2',
                 'C3': 'generator/resnet_v2_50/block2/unit_3/bottleneck_v2',
                 'C4': 'generator/resnet_v2_50/block3/unit_5/bottleneck_v2',
                 'C5': 'generator/resnet_v2_50/block4/unit_3/bottleneck_v2',
                 },
    'resnet101':{
                 'C1': 'generator/resnet_v2_101/conv1',
                 'C2': 'generator/resnet_v2_101/block1/unit_2/bottleneck_v2',
                 'C3': 'generator/resnet_v2_101/block2/unit_3/bottleneck_v2',
                 'C4': 'generator/resnet_v2_101/block3/unit_22/bottleneck_v2',
                 'C5': 'generator/resnet_v2_101/block4',
    },
    'inception-resnet': {
        'C1': 'generator/Conv2d_2b_3x3',
        'C2': 'generator/Conv2d_4a_3x3',
        'C3': 'generator/Mixed_5b',
        'C4': 'generator/Mixed_6a',
        'C5': 'generator/Conv2d_7b_1x1',
    }
}

def dilation_block(name, inputs, input_n, strides=1, padding='SAME', add_bias=True, apply_relu=False, atrous_rate=2):
    with tf.variable_scope(name):
        w_kernel = tf.get_variable(shape=[3,3,input_n,input_n],  name='w')
        if not atrous_rate:
            conv_out = tf.nn.conv2d(inputs, w_kernel, strides, padding)
        else:
            conv_out = tf.nn.atrous_conv2d(inputs, w_kernel, atrous_rate, padding)
        if add_bias:
            w_bias = tf.get_variable(shape=[input_n] ,name='b')
            conv_out = tf.nn.bias_add(conv_out, w_bias)
        if apply_relu:
            conv_out = tf.nn.relu(conv_out)
    return conv_out


def get_base_network(inputs, name='resnet50', weight_decay=1e-6,training=True):
    if name == 'resnet50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_50(inputs, is_training=training)
    elif name == 'inception-resnet':
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=weight_decay)):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(inputs, is_training=training)
    else:
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_101(inputs, is_training=training)
    var_list_restore = {}
    for i in end_points:
        name = i.replace('generator/', '')
        var_list_restore[i] = name
    return var_list_restore, end_points


def build_pyramid(end_points,training, net_name='resnet50',base_fea_n=512):
    pyramid = {}
    pyramid_map = basenet_map[net_name]
    arg_scope = extra_conv_arg_scope()
    with tf.variable_scope('pyramid'):
        with slim.arg_scope(arg_scope):
            p5 = slim.conv2d(end_points[pyramid_map['C5']], base_fea_n, [1, 1], stride=1,
                                        scope='C5')
            pyramid['P5'] = dilation_block('dila_p5',p5,input_n=base_fea_n,atrous_rate=2)
            for i in range(4, 1, -1):
                p, c = pyramid['P%d' % (i + 1)], end_points[pyramid_map['C%d' % (i)]]
                c = slim.dropout(c, keep_prob=0.7, is_training=training)
                up_shape = tf.shape(c)
                p = slim.conv2d(p, base_fea_n, [1, 1], stride=1, scope='P%d' % i)
                p = tf.image.resize_bilinear(p, [up_shape[1], up_shape[2]], name='C%d/upscale' % i)
                c = slim.conv2d(c, base_fea_n, [1, 1], stride=1, scope='C%d' % i, activation_fn=tf.nn.relu)
                p = tf.add(p, c, name='C%d/addition' % i)
                if i < 4:
                    p_skip = pyramid['P%d' % (i + 2)]
                    up_shape = tf.shape(p)
                    p_skip = tf.image.resize_bilinear(p_skip, [up_shape[1], up_shape[2]], name='C%d/upscale' % i)
                    p_skip = slim.conv2d(p_skip, base_fea_n, [3, 3], stride=1, scope='C%d/p_skip_conv_3' % i,
                                         activation_fn=tf.nn.relu)
                    p = tf.add(p, p_skip, name='C%d/addition_skip' % i)
                p_k3 = slim.conv2d(p, base_fea_n, [3, 3], stride=1, scope='C%d/fusion' % i)
                p_k3_dila = dilation_block('dila_fusion_%d'%i,p,input_n=base_fea_n,atrous_rate=2)
                pyramid['P%d' % (i)] = p_k3+p_k3_dila
            return pyramid


def build_output(pyramid,out_size=(96,72),joints_num=24,out_N=4,training=True,base_fea_n=512):
    arg_scope = extra_conv_arg_scope()
    with tf.variable_scope('r_out'):
        with slim.arg_scope(arg_scope):
            with slim.arg_scope([slim.batch_norm],is_training=training) as sc:
              refine_hms = []
              for i in range(2, 6):
                  refine = pyramid['P%d' % (i)]
                  refine1 = dilation_block('dila1_%d' % i, refine, input_n=base_fea_n, atrous_rate=2)
                  refine2 = dilation_block('dila2_%d' % i, refine, input_n=base_fea_n, atrous_rate=3)
                  refine = refine1+refine2
                  for j in range(i - 2):
                      refine = resnet_v2.bottleneck(refine, 256, 128, stride=1)
                  refine = tf.image.resize_bilinear(refine, [out_size[0], out_size[1]],
                                                  name='refine_output%d' % i)
                  refine_hms.append(refine)
              refine_hm = tf.concat(refine_hms, axis=3)
              refine_hm = resnet_v2.bottleneck(refine_hm, 256, 128, stride=1)
              r_out = slim.conv2d(refine_hm, joints_num, [3, 3], stride=1, scope='refine_out',
                                     activation_fn=None)
    output = [None] * out_N
    arg_scope = extra_conv_arg_scope()
    with tf.variable_scope('g_out'):
        with slim.arg_scope(arg_scope):
            for i, name in enumerate(pyramid):
                p = slim.conv2d(pyramid[name], joints_num, [1, 1], stride=1, scope='p%d' % i,
                                activation_fn=None)
                p = tf.image.resize_bilinear(p, [out_size[0], out_size[1]], name='output%d' % i)
                output[i] = p
    return tf.stack(output, axis=1, name='g_out'),r_out

def generator(inputs,training=True,basenet='resnet101',out_size=(96,72),joints_num=24,out_N=4):
    with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
       var_list_restore, end_points = get_base_network(inputs, basenet,training=training)
       pyramid = build_pyramid(end_points,training ,basenet)
       g_out,r_out = build_output(pyramid, out_size=out_size,joints_num=joints_num,out_N=out_N,training=training)
       return g_out,r_out,var_list_restore

if __name__ == '__main__':
    inputs =tf.placeholder(dtype=tf.float32, shape=(1, 384, 288, 3),name='inputs')
    g_out, r_out, var_list_restore=generator(inputs)
    print(g_out)
    print(r_out)
    print(var_list_restore)