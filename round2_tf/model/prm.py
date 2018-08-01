import  tensorflow as tf
import  tensorflow.contrib.slim as slim
import  math



class Conv2d():
    def __init__(self,outnum,ksize,stride,padding):
        self.ksize=ksize
        self.outnum=outnum
        self.stride=stride
        self.padding = padding
    def __call__(self, inputs,name):
        with tf.name_scope(name):
            return slim.conv2d(inputs,self.outnum,self.ksize,self.stride,self.padding,activation_fn=None)

class FractionalMaxPool2d():
    def __init__(self,ratio):
        self.r=ratio

    def __call__(self, inputs,name):
        #with tf.device('/cpu:0'):
        #return slim.max_pool2d(inputs,kernel_size=2)
        #out,_,_=tf.nn.fractional_max_pool(inputs,pooling_ratio=[1,self.r,self.r,1],name=name)
        h =inputs.get_shape().as_list()[1]
        w = inputs.get_shape().as_list()[2]
        s = [int(h*self.r),int(w*self.r)]
        return tf.image.resize_bilinear(inputs, s, name=name)
        #return out



class Upsample():
    def __init__(self,size):
        self.s=size
    def __call__(self, inputs,name):
        return tf.image.resize_bilinear(inputs, self.s, name='upsampling')

class Squential():
    def __init__(self, *args):
        self.Ops=[]
        for o in args:
            self.Ops.append(o)
    def __call__(self, inputs):
        out = self.Ops[0](inputs)
        for op in range(1,len(self.Ops)):
            out = self.Ops[op](out)
        return out


class Pyramid():
    def __init__(self,D,C,inputRes):
        self.D=D
        self.C=C
        self.sc = 2 ** (1 / C)
        for i in range(self.C):
            scaled = 1 / self.sc ** (i + 1)
            setattr(self, 'conv_' + str(i), Conv2d(D, ksize=3, stride=1, padding='SAME'))
            setattr(self, 'SpatialFractionalMaxPooling_' + str(i), FractionalMaxPool2d(ratio=scaled))
            setattr(self, 'SpatialUpSamplingBilinear_' + str(i), Upsample(size=[inputRes[0], inputRes[1]]))

    def __call__(self,inputs,name):
        with tf.name_scope(name):
            pyr = eval('self.SpatialFractionalMaxPooling_' + str(0))(inputs,'pool0')
            pyr = eval('self.conv_' + str(0))(pyr,'conv0')
            pyr = eval('self.SpatialUpSamplingBilinear_' + str(0))(pyr,'up0')

            for i in range(1, self.C):
                x = eval('self.SpatialFractionalMaxPooling_' + str(i))(inputs,'pool'+str(i))
                x = eval('self.conv_' + str(i))(x,'conv'+str(i))
                x = eval('self.SpatialUpSamplingBilinear_' + str(i))(x,'up'+str(i))
                pyr = tf.add(pyr,x)
            return pyr


class pyr_conv_block():
    def __init__(self, numOut, inputRes, type, baseWidth, cardinality, is_train=True):
        self.numOut = numOut
        self.inputRes = inputRes
        self.type = type
        self.baseWidth = baseWidth
        self.cardinality = cardinality
        self.D = math.floor(self.numOut / self.baseWidth)
        self.C = self.cardinality
        self.pyr = Pyramid(self.D,self.C,inputRes=inputRes)
        self.is_train=is_train
    def __call__(self, inputs,name):
        with tf.name_scope(name):
            if self.type != 'no_preact':
                inputs = slim.batch_norm(inputs=inputs,decay=0.997,is_training=self.is_train,activation_fn=tf.nn.relu)
            output = slim.conv2d(inputs,num_outputs=int(self.numOut/2),kernel_size=1,activation_fn=None)
            output = slim.batch_norm(inputs=output,decay=0.997,is_training=self.is_train,activation_fn=tf.nn.relu)
            output = slim.conv2d(output, num_outputs=int(self.numOut / 2), kernel_size=3, activation_fn=None)
            s1 =output
            output = slim.conv2d(inputs, num_outputs=self.D, kernel_size=1, activation_fn=None)
            output = slim.batch_norm(inputs=output, decay=0.997, is_training=self.is_train, activation_fn=tf.nn.relu)
            output = self.pyr(output,'pyr')
            output = slim.batch_norm(inputs=output, decay=0.997, is_training=self.is_train, activation_fn=tf.nn.relu)
            output = slim.conv2d(output, num_outputs=int(self.numOut / 2), kernel_size=1, activation_fn=None)
            #print(s1)

            s2 = output
            #print(s2)
            x = tf.add(s1,s2)
            x = slim.batch_norm(inputs=x, decay=0.997, is_training=self.is_train, activation_fn=tf.nn.relu)
            x = slim.conv2d(x, num_outputs=self.numOut, kernel_size=1, activation_fn=None)
            return x

class   conv_block():
    def __init__(self,numOut,type,is_train):
        self.numOut=numOut
        self.type=type
        self.is_train=is_train
    def __call__(self, inputs,name):
        with tf.name_scope(name):
            # if self.type != 'no_preact':
            #     return  slim.batch_norm(inputs=inputs, decay=0.9, is_training=self.is_train, activation_fn=tf.nn.relu)
            # else:
                out = slim.conv2d(inputs=inputs,num_outputs=int(self.numOut / 2), kernel_size=1,activation_fn=None)
                out = slim.batch_norm(inputs=out, decay=0.997, is_training=self.is_train, activation_fn=tf.nn.relu)
                out = slim.conv2d(inputs=out, num_outputs=int(self.numOut / 2), kernel_size=3, activation_fn=None)
                out = slim.batch_norm(inputs=out, decay=0.997, is_training=self.is_train, activation_fn=tf.nn.relu)
                out = slim.conv2d(inputs=out, num_outputs=self.numOut, kernel_size=1, activation_fn=None)
                return out

def skip_block(inputs,numOut,name,is_train=True):
    with tf.name_scope(name):
        x = slim.batch_norm(inputs=inputs, decay=0.997, is_training=is_train, activation_fn=tf.nn.relu)
        x = slim.conv2d(x, num_outputs=numOut, kernel_size=1, activation_fn=None)
        return x

def pyr_residual(inputs, numOut,  type,  inputRes, baseWidth, cardinality,name,is_train=True):
    with tf.name_scope(name):
        c_net = pyr_conv_block(numOut=numOut, inputRes=inputRes, type=type,
                       baseWidth=baseWidth, cardinality=cardinality,is_train=is_train)
        conv = c_net(inputs,'conv_block')
        skip = skip_block(inputs,numOut,'skip_block',is_train)
        return tf.add(conv,skip)

def ori_residual(inputs, numOut,  type=None,  inputRes=None, baseWidth=None, cardinality=None,name='ori_res',is_train=True):
    with tf.name_scope(name):
        c_net = conv_block(numOut=numOut,  type=type,is_train=is_train)
        conv = c_net(inputs,'conv_block')
        skip = skip_block(inputs,numOut,'skip_block',is_train)
        return tf.add(conv,skip)

def prm_hourglass(inputs,n,f,inputRes,nModules,type,B,C,name,is_train):
    if n>=2:
        resUp = pyr_residual
    else:
        resUp = ori_residual
    if n >= 3:
        resDown = pyr_residual
    else:
        resDown= ori_residual
    with tf.name_scope(name):
        ## upper
        up1 = inputs
        for i in range(nModules):
            #pyr_residual(inputs, numOut,  type,  inputRes, baseWidth, cardinality,name,is_train=True):
            up1 = resUp(up1, f, type, inputRes,  B, C,'resUp_' + str(i), is_train)
        ## lower
        res=  (int(inputRes[0]/2), int(inputRes[1]/2))

        low1 = slim.max_pool2d(inputs, kernel_size=2)
        for i in range(nModules):
            low1 = resDown(low1, f, type, res,  B, C, 'resDown1_' + str(i),is_train)
        if n > 1:
            #(inputs,n,f,inputRes,nModules,type,B,C,name,is_train)
            low2 = prm_hourglass(low1, n - 1, f,res,nModules,type,B,C,name,is_train)
        else:
            low2 = low1
            for i in range(nModules):
                low2 = resDown(low2, f, type, res,  B, C, 'resDown2_' + str(i),is_train)
        low3 = low2
        for i in range(nModules):
            low3 = resDown(low3, f, type, res,  B, C,'resDown3_' + str(i), is_train)
        up2 = tf.image.resize_bilinear(low3, tf.shape(up1)[1:3] * 2, name='upsampling')
        return  tf.add(up1,up2,name='out_hg')


def ori_hourglass(inputs, n, numOut, name='hourglass',is_train=True):
    with tf.name_scope(name):
        # upper 
        up_1 = ori_residual(inputs, numOut, name='up_1',is_train=is_train)
        # lower
        low1 =slim.max_pool2d(inputs, kernel_size=2)
        low1 = ori_residual(low1, numOut, name='low_1',is_train=is_train)
        if n > 0:
            low2 = ori_hourglass(low1, n - 1, numOut, name='low_2',is_train=is_train)
        else:
            low2 = ori_residual(low1, numOut, name='low_2',is_train=is_train)
        low3 = ori_residual(low2, numOut, name='low_3')
        up_2 = tf.image.resize_bilinear(low3, tf.shape(up_1)[1:3] * 2, name='upsampling')
        return tf.add_n([up_2, up_1], name='out_hg')



if __name__=='__main__':

    x = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])
    #y = ori_residual(x,64,'','p1')


    y = prm_hourglass(x, 4, 128, 256, 1, '', 6, 30, 'prm_h1', True)
    print(y)
    # net = Pyramid(6,30,256)
    # x = tf.placeholder(dtype=tf.float32,shape=[None,256,256,3])
    # y = net(x,'p1')
    # print(y)



