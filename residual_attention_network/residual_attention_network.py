# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import tensorflow as tf
import numpy as np

from .basic_layers import ResidualBlock
from .attention_module import AttentionModule,AttentionModule_stage0,AttentionModule_stage1,AttentionModule_stage2,AttentionModule_stage3


class ResidualAttentionNetwork(object):
    """
    Residual Attention Network
    URL: https://arxiv.org/abs/1704.06904
    """
    def __init__(self):
        """
        :param input_shape: the list of input shape (ex: [None, 28, 28 ,3]
        :param output_dim:
        """
        self.input_shape = [-1, 224, 224, 1]
        self.output_dim = 2

        self.attention_module = AttentionModule()
        self.attention_module0 = AttentionModule_stage0()
        self.attention_module1 = AttentionModule_stage1()
        self.attention_module2 = AttentionModule_stage2()
        self.attention_module3 = AttentionModule_stage3()
        self.residual_block = ResidualBlock()

    def f_prop(self, x, is_training=True):
        """
        forward propagation
        :param x: input Tensor [None, row, line, channel]
        :return: outputs of probabilities
        """
        # x = [None, row, line, channel]

        # conv, x -> [None, row, line, 32]
        x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=1, padding='SAME')
        print("conv:",x.shape)
        # max pooling, x -> [None, row, line, 32]
        x = tf.nn.max_pool(x, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='SAME')
        print("max_pool",x.shape)
        # attention module, x -> [None, row, line, 32]
        x = self.attention_module.f_prop(x, input_channels=32, scope="attention_module_1", is_training=is_training)
        print("attention",x.shape)
        # residual block, x-> [None, row, line, 64]
        x = self.residual_block.f_prop(x, input_channels=32, output_channels=64, scope="residual_block_1",
                                       is_training=is_training)
        print("residual",x.shape)
        # max pooling, x -> [None, row, line, 64]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        print("max pool",x.shape)
        # attention module, x -> [None, row, line, 64]
        x = self.attention_module.f_prop(x, input_channels=64, scope="attention_module_2", is_training=is_training)
        print("attention",x.shape)
        # residual block, x-> [None, row, line, 128]
        x = self.residual_block.f_prop(x, input_channels=64, output_channels=128, scope="residual_block_2",
                                       is_training=is_training)
        print("residual",x.shape)
        # max pooling, x -> [None, row/2, line/2, 128]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        print("max pool",x.shape)
        # attention module, x -> [None, row/2, line/2, 64]
        x = self.attention_module.f_prop(x, input_channels=128, scope="attention_module_3", is_training=is_training)
        print("attention",x.shape)
        # residual block, x-> [None, row/2, line/2, 256]
        x = self.residual_block.f_prop(x, input_channels=128, output_channels=256, scope="residual_block_3",
                                       is_training=is_training)
        print("residual",x.shape)
        # max pooling, x -> [None, row/4, line/4, 256]
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        print("max pool",x.shape)
        # residual block, x-> [None, row/4, line/4, 256]
        x = self.residual_block.f_prop(x, input_channels=256, output_channels=256, scope="residual_block_4",
                                       is_training=is_training)
        print("residual",x.shape)
        # residual block, x-> [None, row/4, line/4, 256]
        x = self.residual_block.f_prop(x, input_channels=256, output_channels=256, scope="residual_block_5",
                                       is_training=is_training)
        print("residual",x.shape)
        # residual block, x-> [None, row/4, line/4, 256]
        x = self.residual_block.f_prop(x, input_channels=256, output_channels=256, scope="residual_block_6",
                                       is_training=is_training)
        print("residual",x.shape)
        # average pooling
        x = tf.nn.avg_pool(x, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
        print("avg pool", x.shape)
        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
        print("reshape", x.shape)
        # layer normalization
        x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        # FC, softmax
        y = tf.layers.dense(x, self.output_dim, activation=tf.nn.softmax)
        print("fc", y.shape)

        return y

    def attention_56(self, x, is_training=True):
        # input 224*224
        x = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=2, padding='SAME')
        x = ResidualBlock.batch_norm(x,64)
        print("conv:", x.shape)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        print("max pool", x.shape)
        x = self.residual_block.f_prop(x, input_channels=64, output_channels=256, scope="residual_block_1",
                                       is_training=is_training)
        print("residual", x.shape)
        x = self.attention_module1 .f_prop(x, in_channels=256, out_channels=256, scope="attention_module1")
        print("attention", x.shape)


        x = self.residual_block.f_prop(x, input_channels=256, output_channels=512, scope="residual_block_2",
                                       is_training=is_training, stride=2)
        print("residual", x.shape)
        x = self.attention_module2.f_prop(x, in_channels=512, out_channels=512, scope="attention_module2")
        print("attention", x.shape)


        x = self.residual_block.f_prop(x, input_channels=512, output_channels=1024, scope="residual_block_3",
                                       is_training=is_training, stride=2)
        print("residual", x.shape)
        x = self.attention_module3.f_prop(x, in_channels=1024, out_channels=1024, scope="attention_module3")
        print("attention", x.shape)



        x = self.residual_block.f_prop(x, input_channels=1024, output_channels=2048, scope="residual_block_4",
                                       is_training=is_training, stride=2)
        print("residual", x.shape)
        x = self.residual_block.f_prop(x, input_channels=2048, output_channels=2048, scope="residual_block_5",
                                       is_training=is_training)
        print("residual", x.shape)
        x = self.residual_block.f_prop(x, input_channels=2048, output_channels=2048, scope="residual_block_6",
                                       is_training=is_training)
        print("residual", x.shape)
        x = ResidualBlock.batch_norm(x, 2048)
        x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
        print("avg pool", x.shape)
        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
        # print("reshape", x.shape)
        # layer normalization
        x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        # FC, softmax
        y = tf.layers.dense(x, self.output_dim, activation=tf.nn.softmax)
        print("fc", y.shape)

        return y

    def attention_92(self, x, is_training=True):
        # for input size 224
        x = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=2, padding='SAME')
        x = ResidualBlock.batch_norm(x, 64)

        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        x = self.residual_block.f_prop(x, input_channels=64, output_channels=256, scope="residual_block_1",
                                       is_training=is_training)

        x = self.attention_module1.f_prop(x, in_channels=256, out_channels=256, scope="attention_module1")

        x = self.residual_block.f_prop(x, input_channels=256, output_channels=512, scope="residual_block_2",
                                       is_training=is_training, stride=2)

        x = self.attention_module2.f_prop(x, in_channels=512, out_channels=512, scope="attention_module2_1")
        x = self.attention_module2.f_prop(x, in_channels=512, out_channels=512, scope="attention_module2_2")

        x = self.residual_block.f_prop(x, input_channels=512, output_channels=1024, scope="residual_block_3",
                                       is_training=is_training, stride=2)

        x = self.attention_module3.f_prop(x, in_channels=1024, out_channels=1024, scope="attention_module3_1")
        x = self.attention_module3.f_prop(x, in_channels=1024, out_channels=1024, scope="attention_module3_2")
        x = self.attention_module3.f_prop(x, in_channels=1024, out_channels=1024, scope="attention_module3_3")

        x = self.residual_block.f_prop(x, input_channels=1024, output_channels=2048, scope="residual_block_4",
                                       is_training=is_training, stride=2)

        x = self.residual_block.f_prop(x, input_channels=2048, output_channels=2048, scope="residual_block_5",
                                       is_training=is_training)

        x = self.residual_block.f_prop(x, input_channels=2048, output_channels=2048, scope="residual_block_6",
                                       is_training=is_training)

        x = ResidualBlock.batch_norm(x, 2048)
        x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')

        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
        # # layer normalization
        # x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        # FC, softmax
        y = tf.layers.dense(x, self.output_dim, activation=tf.nn.softmax)

        return y

    def attention_92_448input(self, x, is_training=True):
        # for input size 448
        x = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=2, padding='SAME')
        x = ResidualBlock.batch_norm(x, 64)

        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        x = self.residual_block.f_prop(x, input_channels=64, output_channels=128, scope="residual_block_0",
                                       is_training=is_training)

        x = self.attention_module0.f_prop(x, in_channels=128, out_channels=128, scope="attention_module0")

        x = self.residual_block.f_prop(x, input_channels=128, output_channels=256, scope="residual_block_1",
                                       is_training=is_training, stride=2)

        x = self.attention_module1.f_prop(x, in_channels=256, out_channels=256, scope="attention_module1")

        x = self.residual_block.f_prop(x, input_channels=256, output_channels=512, scope="residual_block_2",
                                       is_training=is_training, stride=2)

        x = self.attention_module2.f_prop(x, in_channels=512, out_channels=512, scope="attention_module2_1")
        x = self.attention_module2.f_prop(x, in_channels=512, out_channels=512, scope="attention_module2_2")

        x = self.residual_block.f_prop(x, input_channels=512, output_channels=1024, scope="residual_block_3",
                                       is_training=is_training, stride=2)

        x = self.attention_module3.f_prop(x, in_channels=1024, out_channels=1024, scope="attention_module3_1")
        x = self.attention_module3.f_prop(x, in_channels=1024, out_channels=1024, scope="attention_module3_2")
        x = self.attention_module3.f_prop(x, in_channels=1024, out_channels=1024, scope="attention_module3_3")

        x = self.residual_block.f_prop(x, input_channels=1024, output_channels=2048, scope="residual_block_4",
                                       is_training=is_training, stride=2)

        x = self.residual_block.f_prop(x, input_channels=2048, output_channels=2048, scope="residual_block_5",
                                       is_training=is_training)

        x = self.residual_block.f_prop(x, input_channels=2048, output_channels=2048, scope="residual_block_6",
                                       is_training=is_training)

        x = ResidualBlock.batch_norm(x, 2048)
        x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')

        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
        # print("reshape", x.shape)
        # layer normalization
        x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
        # FC, softmax
        y = tf.layers.dense(x, self.output_dim, activation=tf.nn.softmax)
        print("fc", y.shape)

        return y










