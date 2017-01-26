#! -*- coding:utf-8 -*-
import os
import sys

import numpy as np
import tensorflow as tf

from util import linear_layer
from batch_normalize import batch_norm
from encoder import Encoder

class Classifier(object):
    def __init__(self, layer_list, name = 'classifier'):
        self.encoder = Encoder(layer_list, '{}_encoder'.format(name))
        self.name_scope = name

    def set_model(self, x, is_training = True):
        with tf.variable_scope(self.name_scope):
            logits = self.encoder.set_model(x, is_training) 

        return tf.nn.softmax(logits)

if __name__ == u'__main__':
    c = Classifier([28 * 28 * 1 , 1200, 600, 10])
    x = tf.placeholder(dtype = tf.float32, shape = [None, 28 * 28 * 1])
    prob = c.set_model(x)
