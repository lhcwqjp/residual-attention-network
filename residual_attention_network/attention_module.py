# -*- coding: utf-8 -*-
"""
attention module of Residual Attention Network
"""

import tensorflow as tf
from .basic_layers import ResidualBlock


class AttentionModule(object):
    """AttentionModuleClass"""
    def __init__(self, p=1, t=2, r=1):
        """
        :param p: the number of pre-processing Residual Units before splitting into trunk branch and mask branch
        :param t: the number of Residual Units in trunk branch
        :param r: the number of Residual Units between adjacent pooling layer in the mask branch
        """
        self.p = p
        self.t = t
        self.r = r

        self.residual_block = ResidualBlock()

    def f_prop(self, input, in_channels, out_channels, scope="attention_module", is_training=True):
        """
        f_prop function of attention module
        :param input: A Tensor. input data [batch_size, height, width, channel]
        :param input_channels: dimension of input channel.
        :param scope: str, tensorflow name scope
        :param is_training: boolean, whether training step or not(test step)
        :return: A Tensor [batch_size, height, width, channel]
        """
        with tf.variable_scope(scope):

            # residual blocks(TODO: change this function)
            with tf.variable_scope("first_residual_blocks"):
                for i in range(self.p):
                    input = self.residual_block.f_prop(input, in_channels, out_channels, scope="num_blocks_{}".format(i), is_training=is_training)

            with tf.variable_scope("trunk_branch"):
                output_trunk = input
                for i in range(self.t):
                    output_trunk = self.residual_block.f_prop(output_trunk, in_channels, out_channels,  scope="num_blocks_{}".format(i), is_training=is_training)

            with tf.variable_scope("soft_mask_branch"):

                with tf.variable_scope("down_sampling_1"):
                    # max pooling
                    k_filter_ = [1, 3, 3, 1]
                    s_filter_ = [1, 2, 2, 1]
                    output_soft_mask = tf.nn.max_pool(input, ksize=k_filter_, strides=s_filter_, padding='SAME')

                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, in_channels, out_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                with tf.variable_scope("skip_connection1"):
                    # TODO(define new blocks)
                    output_skip_connection1 = self.residual_block.f_prop(output_soft_mask, in_channels, out_channels, is_training=is_training)


                with tf.variable_scope("down_sampling_2"):
                    # max pooling
                    k_filter_ = [1, 3, 3, 1]
                    s_filter_ = [1, 2, 2, 1]
                    output_soft_mask = tf.nn.max_pool(output_soft_mask, ksize=k_filter_, strides=s_filter_, padding='SAME')

                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask,  in_channels, out_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                with tf.variable_scope("skip_connection2"):
                    # TODO(define new blocks)
                    output_skip_connection2 = self.residual_block.f_prop(output_soft_mask, in_channels, out_channels,
                                                                         is_training=is_training)

                with tf.variable_scope("down_sampling_3"):
                    # max pooling
                    k_filter_ = [1, 3, 3, 1]
                    s_filter_ = [1, 2, 2, 1]
                    output_soft_mask = tf.nn.max_pool(output_soft_mask, ksize=k_filter_, strides=s_filter_,
                                                      padding='SAME')

                with tf.variable_scope("softmax3_blocks"):
                    for i in range(self.t):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, in_channels, out_channels,
                                                                      scope="num_blocks_{}".format(i),
                                                                      is_training=is_training)


                with tf.variable_scope("up_sampling_1"):
                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, in_channels, out_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                    # interpolation
                    output_soft_mask = tf.keras.layers.UpSampling2D([2, 2])(output_soft_mask)

                # add skip connection
                output_soft_mask += output_skip_connection2

                with tf.variable_scope("up_sampling_2"):
                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, in_channels, out_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                    # interpolation
                    output_soft_mask = tf.keras.layers.UpSampling2D([2, 2])(output_soft_mask)

                # add skip connection
                output_soft_mask += output_skip_connection1

                with tf.variable_scope("up_sampling_3"):
                    for i in range(self.r):
                        output_soft_mask = self.residual_block.f_prop(output_soft_mask, in_channels, out_channels, scope="num_blocks_{}".format(i), is_training=is_training)

                    # interpolation
                    output_soft_mask = tf.keras.layers.UpSampling2D([2, 2])(output_soft_mask)


                with tf.variable_scope("softmax6_blocks "):
                    self.residual_block.batch_norm(output_soft_mask,out_channels,is_training)
                    output_soft_mask = tf.layers.conv2d(output_soft_mask, filters=out_channels, kernel_size=1)
                    self.residual_block.batch_norm(output_soft_mask, out_channels, is_training)
                    output_soft_mask = tf.layers.conv2d(output_soft_mask, filters=out_channels, kernel_size=1)

                    # sigmoid
                    output_soft_mask = tf.nn.sigmoid(output_soft_mask)

            with tf.variable_scope("attention"):
                output = (1 + output_soft_mask) * output_trunk

            with tf.variable_scope("last_residual_blocks"):
                for i in range(self.p):
                    output = self.residual_block.f_prop(output, in_channels, out_channels, scope="num_blocks_{}".format(i), is_training=is_training)

            return output

class AttentionModule_stage0(object):
    # input size is 112*112
    def __init__(self):

        self.residual_block = ResidualBlock()


    def f_prop(self, input, in_channels, out_channels, scope="attention_stage0", is_training=True):
        with tf.variable_scope(scope):
            # 112*112
            x = self.residual_block.f_prop(input,in_channels,out_channels,scope="frist_blocks",is_training=True)
            # trunk
            out_trunk = self.residual_block.f_prop(x,in_channels,out_channels,scope="trunk_blocks1",is_training=True)
            out_trunk = self.residual_block.f_prop(out_trunk, in_channels, out_channels,scope="trunk_blocks2", is_training=True)
            out_mpool1 = tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME')
            # 56*56
            out_softmax1 = self.residual_block.f_prop(out_mpool1,in_channels,out_channels,scope="out_softmax1",is_training=True)
            out_skip1_connection = self.residual_block.f_prop(out_softmax1,in_channels,out_channels,scope="out_skip1_connection",is_training=True)
            out_mpool2 = tf.nn.max_pool(out_softmax1, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME')
            # 28*28
            out_softmax2 = self.residual_block.f_prop(out_mpool2, in_channels, out_channels,scope="out_softmax2", is_training=True)
            out_skip2_connection = self.residual_block.f_prop(out_softmax2, in_channels, out_channels,scope="out_skip2_connection", is_training=True)
            out_mpool3 = tf.nn.max_pool(out_softmax2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 14*14
            out_softmax3 = self.residual_block.f_prop(out_mpool3, in_channels, out_channels,scope="out_softmax3", is_training=True)
            out_skip3_connection = self.residual_block.f_prop(out_softmax3, in_channels, out_channels,scope="out_skip3_connection", is_training=True)
            out_mpool4 = tf.nn.max_pool(out_softmax3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 7*7
            out_softmax4 = self.residual_block.f_prop(out_mpool4, in_channels, out_channels,scope="out_softmax4_1", is_training=True)
            out_softmax4 = self.residual_block.f_prop(out_softmax4, in_channels, out_channels,scope="out_softmax4_2", is_training=True)
            out_interp4 = tf.keras.layers.UpSampling2D((2,2))(out_softmax4) + out_softmax3
            out = out_interp4 + out_skip3_connection
            out_softmax5 = self.residual_block.f_prop(out, in_channels, out_channels,scope="out_softmax5", is_training=True)
            out_interp3 = tf.keras.layers.UpSampling2D((2,2))(out_softmax5) + out_softmax2
            out = out_interp3 + out_skip2_connection
            out_softmax6 = self.residual_block.f_prop(out, in_channels, out_channels,scope="out_softmax6", is_training=True)
            out_interp2 = tf.keras.layers.UpSampling2D((2,2))(out_softmax6) + out_softmax1
            out = out_interp2 + out_skip1_connection
            out_softmax7 = self.residual_block.f_prop(out, in_channels, out_channels,scope="out_softmax7", is_training=True)
            out_interp1 = tf.keras.layers.UpSampling2D((2,2))(out_softmax7) + out_trunk

            with tf.variable_scope("out_softmax8"):
                out_interp1 = self.residual_block.batch_norm(out_interp1, out_channels, is_training)
                out_interp1 = tf.layers.conv2d(out_interp1, filters=out_channels, kernel_size=1)
                out_interp1 = self.residual_block.batch_norm(out_interp1, out_channels, is_training)
                out_interp1 = tf.layers.conv2d(out_interp1, filters=out_channels, kernel_size=1)
                # sigmoid
                out_softmax8 = tf.nn.sigmoid(out_interp1)

            with tf.variable_scope("attention"):
                output = (1 + out_softmax8) * out_trunk

            out_last = self.residual_block.f_prop(output, in_channels, out_channels,scope="out_last", is_training=True)

            return out_last

class AttentionModule_stage1(object):
    # input size is 56*56
    def __init__(self):

        self.residual_block = ResidualBlock()

    def f_prop(self, input, in_channels, out_channels, scope="attention_stage1", is_training=True):
        with tf.variable_scope(scope):
            # 56*56
            x = self.residual_block.f_prop(input, in_channels, out_channels,scope="first_blocks", is_training=True)
            # trunk
            out_trunk = self.residual_block.f_prop(x, in_channels, out_channels,scope="trunk1", is_training=True)
            out_trunk = self.residual_block.f_prop(out_trunk, in_channels, out_channels,scope="trunk2", is_training=True)
            out_mpool1 = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 28*28
            out_softmax1 = self.residual_block.f_prop(out_mpool1, in_channels, out_channels,scope="out_softmax1", is_training=True)
            out_skip1_connection = self.residual_block.f_prop(out_softmax1, in_channels, out_channels,scope="skip1", is_training=True)
            out_mpool2 = tf.nn.max_pool(out_softmax1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 14*14
            out_softmax2 = self.residual_block.f_prop(out_mpool2, in_channels, out_channels,scope="out_softmax2", is_training=True)
            out_skip2_connection = self.residual_block.f_prop(out_softmax2, in_channels, out_channels,scope="skip2", is_training=True)
            out_mpool3 = tf.nn.max_pool(out_softmax2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 7*7
            out_softmax3 = self.residual_block.f_prop(out_mpool3, in_channels, out_channels,scope="out_softmax3_1", is_training=True)
            out_softmax3 = self.residual_block.f_prop(out_softmax3, in_channels, out_channels,scope="out_softmax3_2", is_training=True)
            # up
            out_interp3 = tf.keras.layers.UpSampling2D(size=(2,2))(out_softmax3) + out_softmax2
            out = out_interp3 + out_skip2_connection
            out_softmax4 = self.residual_block.f_prop(out, in_channels, out_channels,scope="out_softmax4", is_training=True)
            # up
            out_interp2 = tf.keras.layers.UpSampling2D(size=(2,2))(out_softmax4) + out_softmax1
            out = out_interp2 + out_skip1_connection
            # up
            out_softmax5 = self.residual_block.f_prop(out, in_channels, out_channels,scope="out_softmax5", is_training=True)
            out_interp1 = tf.keras.layers.UpSampling2D(size=(2,2))(out_softmax5) + out_trunk

            with tf.variable_scope("out_softmax6"):
                out_interp1 = self.residual_block.batch_norm(out_interp1, out_channels, is_training)
                out_interp1 = tf.layers.conv2d(out_interp1, filters=out_channels, kernel_size=1)
                out_interp1 = self.residual_block.batch_norm(out_interp1, out_channels, is_training)
                out_interp1 = tf.layers.conv2d(out_interp1, filters=out_channels, kernel_size=1)
                # sigmoid
                out_softmax6 = tf.nn.sigmoid(out_interp1)

            with tf.variable_scope("attention"):
                output = (1 + out_softmax6) * out_trunk

            out_last = self.residual_block.f_prop(output, in_channels, out_channels,scope="out_last", is_training=True)

            return out_last

class AttentionModule_stage2(object):
    # input size is 28*28
    def __init__(self):

        self.residual_block = ResidualBlock()

    def f_prop(self, input, in_channels, out_channels, scope="attention_stage2", is_training=True):
        with tf.variable_scope(scope):
            # 28*28
            x = self.residual_block.f_prop(input, in_channels, out_channels,scope="first_block", is_training=True)
            # trunk
            out_trunk = self.residual_block.f_prop(x, in_channels, out_channels,scope="trunk_1", is_training=True)
            out_trunk = self.residual_block.f_prop(out_trunk, in_channels, out_channels,scope="trunk_2", is_training=True)
            out_mpool1 = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 14*14
            out_softmax1 = self.residual_block.f_prop(out_mpool1, in_channels, out_channels,scope="out_softmax1", is_training=True)
            out_skip1_connection = self.residual_block.f_prop(out_softmax1, in_channels, out_channels,scope="skip1", is_training=True)
            out_mpool2 = tf.nn.max_pool(out_softmax1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 7*7
            out_softmax2 = self.residual_block.f_prop(out_mpool2, in_channels, out_channels,scope="out_softmax2_1", is_training=True)
            out_softmax2 = self.residual_block.f_prop(out_softmax2, in_channels, out_channels,scope="out_softmax2_2", is_training=True)
            # up
            out_interp2 = tf.keras.layers.UpSampling2D(size=(2,2))(out_softmax2) + out_softmax1
            out = out_interp2 + out_skip1_connection
            out_softmax3 = self.residual_block.f_prop(out, in_channels, out_channels,scope="out_softmax3", is_training=True)
            # up
            out_interp1 = tf.keras.layers.UpSampling2D(size=(2,2))(out_softmax3) + out_trunk


            with tf.variable_scope("out_softmax4"):
                out_interp1 = self.residual_block.batch_norm(out_interp1, out_channels, is_training)
                out_interp1 = tf.layers.conv2d(out_interp1, filters=out_channels, kernel_size=1)
                out_interp1 = self.residual_block.batch_norm(out_interp1, out_channels, is_training)
                out_interp1 = tf.layers.conv2d(out_interp1, filters=out_channels, kernel_size=1)
                # sigmoid
                out_softmax4 = tf.nn.sigmoid(out_interp1)

            with tf.variable_scope("attention"):
                output = (1 + out_softmax4) * out_trunk

            out_last = self.residual_block.f_prop(output, in_channels, out_channels,scope="out_last", is_training=True)

            return out_last


class AttentionModule_stage3(object):
    # input size is 14*14
    def __init__(self):

        self.residual_block = ResidualBlock()

    def f_prop(self, input, in_channels, out_channels, scope="attention_stage3", is_training=True):
        with tf.variable_scope(scope):
            # 14*14
            x = self.residual_block.f_prop(input, in_channels, out_channels,scope="first_block", is_training=True)
            # trunk
            out_trunk = self.residual_block.f_prop(x, in_channels, out_channels,scope="trunk1", is_training=True)
            out_trunk = self.residual_block.f_prop(out_trunk, in_channels, out_channels,scope="trunk2", is_training=True)
            out_mpool1 = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 7*7
            out_softmax1 = self.residual_block.f_prop(out_mpool1, in_channels, out_channels,scope="out_softmax1_1", is_training=True)
            out_softmax1 = self.residual_block.f_prop(out_softmax1, in_channels, out_channels,scope="out_softmax1_2", is_training=True)
            # up
            out_interp1 = tf.keras.layers.UpSampling2D(size=(2,2))(out_softmax1) + out_trunk

            with tf.variable_scope("out_softmax2"):
                out_interp1 = self.residual_block.batch_norm(out_interp1, out_channels, is_training)
                out_interp1 = tf.layers.conv2d(out_interp1, filters=out_channels, kernel_size=1)
                out_interp1 = self.residual_block.batch_norm(out_interp1, out_channels, is_training)
                out_interp1 = tf.layers.conv2d(out_interp1, filters=out_channels, kernel_size=1)
                # sigmoid
                out_softmax2 = tf.nn.sigmoid(out_interp1)

            with tf.variable_scope("attention"):
                output = (1 + out_softmax2) * out_trunk

            out_last = self.residual_block.f_prop(output, in_channels, out_channels,scope="out_last", is_training=True)

            return out_last
