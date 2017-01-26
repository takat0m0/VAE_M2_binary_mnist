#! -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import cv2
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

def get_dim(target):
    dim = 1
    for d in target.get_shape()[1:].as_list():
        dim *= d
    return dim

def get_figs(dir_name):
    ret = []
    for file_name in os.listdir(dir_name):
        ret.append(cv2.imread(os.path.join(dir_name, file_name))/255.0)
        
    return np.asarray(ret, dtype = np.float32)

def dump_figs(imgs, dir_name):
    for i, img in enumerate(imgs):
        cv2.imwrite(os.path.join(dir_name, '{}.jpg'.format(i)), img * 255)

