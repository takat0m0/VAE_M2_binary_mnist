#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from util import linear_layer
from batch_normalize import batch_norm

# almost same as encoder.
# but we do not use encoder class here
# because these are different when convlutional neural network

class Decoder(object):
    def __init__(self, layer_list, name = 'decoder'):
        self.layer_list = layer_list
        self.name_scope = name

    def set_model(self, z, y, is_training = True):
        h = tf.concat(1, [z, y])
        num_layers = len(self.layer_list) - 1
        
        with tf.variable_scope(self.name_scope):
            for i, (in_dim, out_dim) in enumerate(zip(self.layer_list, self.layer_list[1:])):
                h = linear_layer(i, h, in_dim, out_dim)
                if i != num_layers - 1:
                    h = batch_norm(h, i, is_training)
                    h = tf.nn.relu(h)
        # return sigmoid because we here consider binary-mnist
        return tf.nn.sigmoid(h)

if __name__ == u'__main__':
    d = Decoder([100 + 10, 600, 1200, 28 * 28 * 1])
    z = tf.placeholder(dtype = tf.float32, shape = [None, 100])
    y = tf.placeholder(dtype = tf.float32, shape = [None, 10])
    sigmoided = d.set_model(z, y)
