import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from depthoptic_multi_utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, img1, depth1, img2, depth2, optic, reuse_variables=None):
        self.input_img1 = img1
        self.input_img2 = img2
        self.gt_depth1  = depth1
        self.gt_depth2  = depth2
        self.raw_gt_optic = optic
        tmp1, tmp2 = tf.split(optic, num_or_size_splits=2, axis=1)               #(-1, 2, height, width)
        tmp1 = tf.reshape(tmp1, (-1, FLAGS.input_height, FLAGS.input_width, 1))
        tmp2 = tf.reshape(tmp2, (-1, FLAGS.input_height, FLAGS.input_width, 1))
        self.gt_optic = tf.negative(tf.concat([tmp1, tmp2], axis=3))                          #(-1, height, width, 2)
        
        self.reuse_variables  = reuse_variables

        self.build_model()
        self.build_outputs()
        self.build_losses()
        self.build_errors()

        if FLAGS.mode == 'test':
            return

        self.build_summaries()

# ================= Math Function =================== #
    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx
    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy
    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio]) 
    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h / ratio
            nw = w / ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def get_depth(self, x):
        return self.conv(x, 2, 3, 1, tf.nn.sigmoid)

    def get_optic(self, x):
        return self.conv(x, 2, 3, 1, tf.nn.leaky_relu)

# ================= Layer Function =================== #
  #  -----------Pooling Layer------------- #
    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)
  #  -----------Normal Conv Layer------------- #
    # single conv layer
    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)
    # multiple conv layers (for vgg)
    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2
    # residual conv layer (skip/ for ResNet)
    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)
    # multiple residual layers
    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

  #  -----------Upsample Conv Layer------------- #
    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv
    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

  #  -----------Dilated Layer------------- #
    # single dilated layer
    def atrousconv(self, x, num_out_layers, kernel_size,stride=1, rate=1, activation_fn=tf.nn.elu):
        return slim.convolution(x, num_out_layers, kernel_size, stride=stride , rate=rate, activation_fn=activation_fn)
                       #       (inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    # residual dilated conv layers (supplent: resconv)
    def dilated_resconv(self, x, num_layers,stride, rate): # 1 ResBlock
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv      (x,       num_layers/2, 1,      1)
        conv2 = self.atrousconv(conv1,   num_layers/2, 3, stride=stride,  rate=rate)
        conv3 = self.conv      (conv2,   num_layers,1, 1,   None)

        if do_proj:  
            shortcut = self.conv(x, num_layers, 1, stride, None)
            #shortcut = self.conv(x, num_layers, 1, stride, None)
        else:  
            shortcut = x 
        return tf.nn.elu(conv3 + shortcut)
    # multiple residual dilated layer 
    def dilated_resblock(self, x, num_layers, num_blocks, rate): # num_blocks ResBlock
        out = x
        for i in range(num_blocks - 1):
            out = self.dilated_resconv(out, num_layers, 1, rate)
        out = self.dilated_resconv(out, num_layers, 1, rate)
        return out

# ======================= Model ====================== #
    def build_drn(self, input1, input2, model_reuse=False):
        conv   = self.conv
        upconv = self.deconv
        input = tf.concat([input1, input2], axis=3)                            #(-1, 128, 256, 6)

        with tf.variable_scope('encoder', reuse=model_reuse):
            conv1  = self.atrousconv(input,16, 7, stride=1, rate=1)            # H,   16,  7
            conv2  = self.dilated_resblock(conv1,  16, 1, 1)
            pool1  = self.maxpool(conv2, 2)                                    # H/2, 16,  ?
            conv3  = self.dilated_resblock(pool1,  32, 1, 1)       
            pool2  = self.maxpool(conv3, 2)                                    # H/4, 32,  ?
            conv4  = self.dilated_resblock(pool2,  64, 1, 1)       
            pool3  = self.maxpool(conv4, 2)                                    # H/8, 64,  ?

            conv5  = self.dilated_resblock(pool3,  128, 2, 1)                  # H/8, 128, 64
            conv6  = self.dilated_resblock(conv5,  256, 2, 2)                  # H/8, 256, 112
            conv7  = self.dilated_resblock(conv6,  512, 2, 4)                  # H/8, 512, 216
            conv8  = self.atrousconv(conv7,        512, 3, rate=2)             # H/8, 512, 248
            conv9  = self.atrousconv(conv8,        512, 3, rate=2)             # H/8, 512, 280
            conv10 = self.atrousconv(conv9,        512, 3, rate=1)             # H/8, 512, 296
            conv11 = self.atrousconv(conv10,       512, 3, rate=1)             # H/8, 512, 312

        with tf.variable_scope('skips',reuse=model_reuse):
            skip1 = conv2                                                      # H
            skip2 = conv3                                                      # H/2 
            skip3 = conv4                                                      # H/4

        with tf.variable_scope('decoder',reuse=model_reuse):
            depth4  = self.get_depth(conv11)                                  # H/8
            optic4  = self.get_optic(conv11)
            udepth4 = self.upsample_nn(depth4, 2)                               # H/4
            upconv3 = upconv(conv11,               64, 3, 2)                   # H/4
            concat3 = tf.concat([upconv3, skip3, udepth4], 3)                   # H/4
            iconv3  = conv(concat3,                64, 3, 1)                   # H/4
            depth3  = self.get_depth(iconv3)                                   # H/4
            optic3  = self.get_optic(iconv3)

            udepth3 = self.upsample_nn(depth3, 2)                               # H/2
            upconv2 = upconv(iconv3,               32, 3, 2)                   # H/2
            concat2 = tf.concat([upconv2, skip2, udepth3], 3)                   # H/2
            iconv2  = conv(concat2,                32, 3, 1)                   # H/2
            depth2  = self.get_depth(iconv2)                                   # H/2
            optic2  = self.get_optic(iconv2)

            udepth2 = self.upsample_nn(depth2, 2)                               # H
            upconv1 = upconv(iconv2,               16, 3, 2)                   # H
            concat1 = tf.concat([upconv1, skip1, udepth2], 3)                   # H
            iconv1  = conv(concat1,                16, 3, 1)                   # H
            depth1  = self.get_depth(iconv1)                                   # H
            optic1  = self.get_optic(iconv1)
            #print(depth1.shape)    (-1, 128, 256, 2)
            #print(optic1.shape)    (-1, 128, 256, 2)
            return depth1, depth2, depth3, depth4, optic1, optic2, optic3, optic4

# ================================================== #
    # --------- Divide Variable -------- #
    def build_var_list(self):
        with tf.name_scope('Variable_seperation'):
            total_vars = tf.trainable_variables()

            self.ae_vars     = [var for var in total_vars if '/autoencoder/' in var.name]
            self.e_vars      = [var for var in total_vars if '/encoder/' in var.name]
            self.d_vars      = [var for var in total_vars if '/decoder/' in var.name]

            self.disnet_vars = self.e_vars + self.d_vars

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):
                self.pre_depth1, self.pre_depth2, self.pre_depth3, self.pre_depth4, \
                    self.pre_optic1, self.pre_optic2, self.pre_optic3, self.pre_optic4  =\
                        self.build_drn(self.input_img1, self.input_img2, model_reuse=False)

    def build_outputs(self):
        self.build_var_list()
        self.input_img1_pyramid = self.scale_pyramid(self.input_img1,  4)
        self.input_img2_pyramid = self.scale_pyramid(self.input_img2,  4)
        self.gt_depth1_pyramid  = self.scale_pyramid(self.gt_depth1,   4)
        self.gt_depth2_pyramid  = self.scale_pyramid(self.gt_depth2,   4)
        self.gt_optic_pyramid   = self.scale_pyramid(self.gt_optic,    4)

        self.pre_depth1_1, self.pre_depth2_1 = tf.split(self.pre_depth1, num_or_size_splits=2, axis=3)
        self.pre_depth1_2, self.pre_depth2_2 = tf.split(self.pre_depth2, num_or_size_splits=2, axis=3)
        self.pre_depth1_3, self.pre_depth2_3 = tf.split(self.pre_depth3, num_or_size_splits=2, axis=3)
        self.pre_depth1_4, self.pre_depth2_4 = tf.split(self.pre_depth4, num_or_size_splits=2, axis=3)

        self.pre_depth1_pyramid = [self.pre_depth1_1, self.pre_depth1_2, self.pre_depth1_3, self.pre_depth1_4]
        self.pre_depth2_pyramid = [self.pre_depth2_1, self.pre_depth2_2, self.pre_depth2_3, self.pre_depth2_4]
        self.pre_optic_pyramid  = [self.pre_optic1, self.pre_optic2, self.pre_optic3, self.pre_optic4]

        self.pre_depth2_1_optic = tf_warp(self.pre_depth1_1, self.raw_gt_optic, FLAGS.input_height, FLAGS.input_width)

    def pyramid_loss(self, pre_pyramid, gt_pyramid):
        l1                      = [tf.abs(pre_pyramid[i] - gt_pyramid[i]) for i in range(4)]
        l1_loss                 = [tf.reduce_mean(l) for l in l1]    
        diff                    = [(pre_pyramid[i] - gt_pyramid[i]) for i in range(4)]
        first_term              = [tf.reduce_mean(tf.square(ln)) for ln in diff]
        second_term             = [tf.square(tf.reduce_sum(ln))/tf.square(tf.cast(tf.size(ln), tf.float32)) for ln in diff]
        scalar_inv_loss         = [(first_term[i]  - 0.5 * second_term[i]) for i in range(4)]
        loss                    = tf.add_n(scalar_inv_loss)
        return loss

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            self.consistency_loss         = tf.reduce_mean(tf.abs(self.pre_depth2_1_optic - self.pre_depth2_1))
            self.depth1_loss              = self.pyramid_loss(self.pre_depth1_pyramid, self.gt_depth1_pyramid)
            self.depth2_loss              = self.pyramid_loss(self.pre_depth2_pyramid, self.gt_depth2_pyramid)
            self.depth_loss               = self.depth1_loss + self.depth2_loss
            self.optic_loss               = self.pyramid_loss(self.pre_optic_pyramid, self.gt_optic_pyramid)
            self.total_loss               = self.optic_loss

        self.train_op = tf.train.AdamOptimizer(FLAGS.init_lr).minimize(self.total_loss)
            
    def build_errors(self):
        with tf.variable_scope('errors', reuse=self.reuse_variables):
            self.depth1_l1     = tf.reduce_mean(tf.abs(self.pre_depth1_pyramid[0]    - self.gt_depth1_pyramid[0]))
            self.depth2_l1     = tf.reduce_mean(tf.abs(self.pre_depth2_pyramid[0]    - self.gt_depth2_pyramid[0]))
            self.depth1_l2     = tf.reduce_mean(tf.square(self.pre_depth1_pyramid[0] - self.gt_depth1_pyramid[0]))
            self.depth2_l2     = tf.reduce_mean(tf.square(self.pre_depth2_pyramid[0] - self.gt_depth2_pyramid[0]))
            
            self.depth_l1      = (self.depth1_l1 + self.depth2_l1)/2
            self.depth_l2      = (self.depth1_l2 + self.depth2_l2)/2
            self.depth_consist = tf.reduce_mean(tf.abs(self.pre_depth2_1_optic       - self.pre_depth2_pyramid[0]))
            
    def of_to_rgb(self, optic_flow):
        optic_r, optic_g = tf.split(optic_flow, num_or_size_splits=2, axis=3)
        extra_b = tf.zeros(shape=tf.shape(optic_r), dtype=tf.float32)
        return tf.concat([optic_r, optic_g, extra_b], axis=3)
    
    def build_summaries(self):
        self.pre_optic_rgb  = self.of_to_rgb(self.pre_optic_pyramid[0])
        self.gt_optic_rgb   = self.of_to_rgb(self.gt_optic_pyramid[0])

        with tf.device('/cpu:0'):
            tf.summary.image('input1',         self.input_img1)
            tf.summary.image('pre_depth1',     self.pre_depth1_pyramid[0])
            tf.summary.image('gt_depth1',      self.gt_depth1_pyramid[0])
            tf.summary.image('input2',         self.input_img2)
            tf.summary.image('pre_depth2',     self.pre_depth2_pyramid[0])
            tf.summary.image('gt_depth2',      self.gt_depth2_pyramid[0])
            
            tf.summary.image('pre_optic',      self.pre_optic_rgb)
            tf.summary.image('gt_optic',       self.gt_optic_rgb)
            
            tf.summary.scalar('total_loss',    self.total_loss)
            tf.summary.scalar('depth_loss',    self.depth_loss)
            tf.summary.scalar('optic_loss',    self.optic_loss)
            tf.summary.scalar('consist_loss',  self.consistency_loss)
            self.merge_op = tf.summary.merge_all()
