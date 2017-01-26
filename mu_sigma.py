#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from util import linear_layer
from batch_normalize import batch_norm
from encoder import Encoder

class MuSigmaEncoder(object):
    def __init__(self, layer_list, z_dim, name = 'mu_sigma'):
        self.encoder = Encoder(layer_list, '{}_encoder'.format(name))
        self.z_dim = z_dim
        self.name_scope = name

    def set_model(self, x, y, is_training = True):
        h = tf.concat(1, [x, y])
        with tf.variable_scope(self.name_scope):
            h = self.encoder.set_model(h, is_training)
            in_dim = h.get_shape().as_list()[-1]
            out_dim = self.z_dim

            mu = linear_layer('mu', h, in_dim, out_dim)

            sigma = tf.exp(linear_layer('sigma', h, in_dim, out_dim))

        return mu, sigma

if __name__ == u'__main__':
    musigma = MuSigmaEncoder([28 * 28 * 1 + 10, 1200, 600], 100)
    x = tf.placeholder(dtype = tf.float32, shape = [None, 28 * 28 * 1])
    y = tf.placeholder(dtype = tf.float32, shape = [None, 10])

    mu, sigma = musigma.set_model(x, y)
