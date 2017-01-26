#! -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np

def get_weights(name, shape, stddev, trainable = True):
    return tf.get_variable('weights{}'.format(name), shape,
                           initializer = tf.random_normal_initializer(stddev = stddev),
                           trainable = trainable)

def get_biases(name, shape, value, trainable = True):
    return tf.get_variable('biases{}'.format(name), shape,
                           initializer = tf.constant_initializer(value),
                           trainable = trainable)

def linear_layer(name, inputs, in_dim, out_dim):
    weights = get_weights(name, [in_dim, out_dim], 1/tf.sqrt(float(in_dim)))
    biases = get_biases(name, [out_dim], 0.0)
    h = tf.matmul(inputs, weights) + biases
    return h
