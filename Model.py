#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from mu_sigma import MuSigmaEncoder
from decoder import Decoder
from classifier import Classifier


def _get_one_hot(target_index, num_batch, num_dim):
    indices = [[_, target_index] for _ in range(num_batch)]
    values = [1.0] * num_batch 
    ret = tf.sparse_tensor_to_dense(
        tf.SparseTensor( indices=indices, values=values, shape=[num_batch, num_dim] ), 0.0 )
    return ret

class Model(object):
    def __init__(self, x_dim, y_dim, z_dim, batch_size):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.classifier = Classifier([x_dim, 1200, 600, y_dim])
        self.ms_encoder = MuSigmaEncoder([x_dim + y_dim, 1200, 600], z_dim)
        self.decoder = Decoder([z_dim + y_dim, 600, 1200,  x_dim])

        self.lr = 0.0005
        
    def set_model(self):

        self.x = tf.placeholder(dtype = tf.float32, shape = [None, self.x_dim])
        self.x_unlabeled = tf.placeholder(dtype = tf.float32, shape = [None, self.x_dim])
        self.y = tf.placeholder(dtype = tf.float32, shape = [None, self.y_dim])

        # labeled model
        obj = self._set_label_model()

        # share variable
        tf.get_variable_scope().reuse_variables()
        
        # unlabeled model
        obj += self._set_unlabel_model()

        # set optimizer
        self.train = tf.train.AdamOptimizer(self.lr).minimize(obj)
        
        # for usage
        self.prob = self.classifier.set_model(self.x, False)
        self.mu, _ = self.ms_encoder.set_model(self.x, self.y, False)

        self.z_const = tf.placeholder(dtype = tf.float32, shape = [None, self.z_dim])
        self.outputs = self.decoder.set_model(self.z_const, self.y, False)

        
    def _set_label_model(self, is_training = True):

        # encode, latent, decode
        mu, sigma = self.ms_encoder.set_model(self.x, self.y, is_training)
        eps = np.random.randn(self.batch_size, self.z_dim)
        z = mu + sigma * eps                
        auto_encoded = self.decoder.set_model(z, self.y, is_training)

        # auto encode loss
        obj = -tf.reduce_sum(self.x * tf.log(1.0e-4 + auto_encoded) + (1 - self.x) * tf.log(1.0e-4 + 1 - auto_encoded), 1)

        # kl
        obj += tf.reduce_sum(mu * mu/2.0 - tf.log(sigma) + sigma * sigma/2.0, 1)
        obj = tf.reduce_mean(obj)
        
        # classifier loss
        prob = self.classifier.set_model(self.x, is_training)
        obj += -tf.reduce_sum(self.y * tf.log(1.0e-4 + prob) +
                              (1 - self.y) * tf.log(1.0e-4 + 1 - prob), 1)
        
        return obj
        
    def _set_unlabel_model(self, is_training = True):

        # encode, latent, decode
        for i in range(self.y_dim):
            tmp_y = _get_one_hot(i, self.batch_size, self.y_dim)
            mu, sigma = self.ms_encoder.set_model(self.x_unlabeled, tmp_y, is_training)
            eps = np.random.randn(self.batch_size, self.z_dim)
            z = mu + sigma * eps
            auto_encoded = self.decoder.set_model(z, tmp_y, is_training)
            
            # auto encode loss
            obj = -tf.reduce_sum(self.x_unlabeled * tf.log(1.0e-4 + auto_encoded) +
                                 (1 - self.x_unlabeled) * tf.log(1.0e-4 + 1 - auto_encoded), 1)
            
            # kl
            obj += tf.reduce_sum(mu * mu/2.0 - tf.log(sigma) + sigma * sigma/2.0, 1)
            
            # prior y
            hoge = 1.0/10 * tf.ones_like(tmp_y)
            obj += -tf.reduce_sum(tmp_y * tf.log(1.0e-4 + hoge) +
                                  (1 - tmp_y) * tf.log(1.0e-4 + 1 - hoge), 1)
            
            #obj = tf.expand_dims(tf.reduce_mean(obj), 1)
            obj = tf.expand_dims(obj, 1)

            if i == 0:
                objs = tf.identity(obj)
            else:
                objs = tf.concat(1, [objs, obj])

        # total_loss
        prob = self.classifier.set_model(self.x_unlabeled, is_training)
        total_obj = tf.reduce_sum(tf.reduce_sum(tf.mul(prob, objs), 1))

        # classifier loss
        total_obj += -tf.reduce_sum(prob * tf.log(1.0e-4 + prob) +
                                    (1 - prob) * tf.log(1.0e-4 + 1 - prob), 1)
                
        return total_obj

    def training(self, sess, x_labeled, y_labeled, x_unlabeled):
        sess.run(self.train,
                 feed_dict = {self.x: x_labeled,
                              self.y: y_labeled,
                              self.x_unlabeled: x_unlabeled})
        
    def get_prob(self, sess, x):
        return sess.run(self.prob, feed_dict = {self.x: x})
    
    def encode(self, sess, x, y):
        return sess.run(self.mu,
                        feed_dict = {self.x: x,
                                     self.y: y})
    
    def generate(self, sess, z, y):
        ret = sess.run(self.outputs,
                       feed_dict = {self.z_const:z,
                                    self.y: y})
        return ret

if __name__ == u'__main__':

    model = Model(x_dim = 28 * 28 * 1, y_dim = 10, z_dim = 100, batch_size = 100)
    obj = model.set_model()
