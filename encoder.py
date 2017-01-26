#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from util import linear_layer
from batch_normalize import batch_norm

class Encoder(object):
    def __init__(self, layer_list, name = 'encoder'):
        self.layer_list = layer_list
        self.name_scope = name

    def set_model(self, x, is_training = True):
        h = x
        num_layers = len(self.layer_list) - 1
        
        with tf.variable_scope(self.name_scope):
            for i, (in_dim, out_dim) in enumerate(zip(self.layer_list, self.layer_list[1:])):
                h = linear_layer(i, h, in_dim, out_dim)
                if i != num_layers - 1:
                    h = batch_norm(h, i, is_training)
                    h = tf.nn.relu(h)
        return h

