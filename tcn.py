#coding: utf-8
'''
Author: Weiping Song
Time: April 24, 2018
'''
import tensorflow as tf
from wnconv1d import wnconv1d

class TemporalConvNet(object):
    def __init__(self, num_channels, stride=1, kernel_size=2, dropout=0.2):
        self.kernel_size=kernel_size
        self.stride = stride
        self.num_levels = len(num_channels)
        self.num_channels = num_channels
        self.dropout = dropout
    
    def __call__(self, inputs):
        inputs_shape = inputs.get_shape().as_list()
        outputs = [inputs]
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = inputs_shape[-1] if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            output = self._TemporalBlock(outputs[-1], in_channels, out_channels, self.kernel_size, 
                                    self.stride, dilation=dilation_size, padding=(self.kernel_size-1)*dilation_size, 
                                    dropout=self.dropout, level=i)
            outputs.append(output)

        return outputs[-1]

    def _TemporalBlock(self, value, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, level=0):
        padded_value1 = tf.pad(value, [[0,0], [padding,0], [0,0]])
        self.conv1 = wnconv1d(inputs=padded_value1,
                                    filters=n_outputs,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding='valid',
                                    dilation_rate=dilation,
                                    activation=None,
                                    weight_norm=True, #default is false.
                                    kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='layer'+str(level)+'_conv1')
        self.output1 = tf.nn.dropout(tf.nn.relu(self.conv1), keep_prob=1-dropout)

        padded_value2 = tf.pad(self.output1, [[0,0], [padding,0], [0,0]])
        self.conv2 = wnconv1d(inputs=padded_value2,
                                    filters=n_outputs,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding='valid',
                                    dilation_rate=dilation,
                                    activation=None,
                                    weight_norm=True, #default is False.
                                    kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='layer'+str(level)+'_conv2')
        self.output2 = tf.nn.dropout(tf.nn.relu(self.conv2), keep_prob=1-dropout)

        if n_inputs != n_outputs:
            res_x = tf.layers.conv1d(inputs=value,
                                    filters=n_outputs,
                                    kernel_size=1,
                                    activation=None,
                                    kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='layer'+str(level)+'_conv')
        else:
            res_x = value
        return tf.nn.relu(res_x + self.output2)
