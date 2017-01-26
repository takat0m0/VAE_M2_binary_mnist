#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Model import Model
from make_fig import get_batch    

def _zero_one(target):
    return 0 if target < 0.5 else 1

if __name__ == u'__main__':

    labeled_file_name = 'labeled.csv'
    unlabeled_file_name = 'unlabeled.csv'
    
    dump_dir = 'sample_result'
    if os.path.exists(dump_dir) == False:
        os.mkdir(dump_dir)
        
    # parameter
    batch_size = 100
    z_dim = 20
    epoch_num = 100
    
    # make model
    model = Model(x_dim = 28 * 28 * 1, y_dim = 10, z_dim = z_dim, batch_size = batch_size)
    model.set_model()

    # get labeled data
    with open(labeled_file_name, 'r') as f_obj:
        y_labeled, x_labeled = get_batch(f_obj, 100)
        
    # get unlabeld data
    num_unlabeled_data = sum(1 for _ in open(unlabeled_file_name))
    with open(unlabeled_file_name, 'r') as f_obj:
        _, tmp = get_batch(f_obj, num_unlabeled_data)
        u_data = np.asarray(tmp)
        
    num_one_epoch =  num_unlabeled_data//batch_size
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(epoch_num):
            for step in range(num_one_epoch):
                x_unlabeled = u_data[step * batch_size: (step + 1) * batch_size]
                model.training(sess, x_labeled, y_labeled, x_unlabeled)
                    
                if step%1000 == 0:
                    print(epoch)
                    z1 = model.encode(sess, [x_labeled[1]], [y_labeled[1]])
                    tmp = model.generate(sess, z1, [y_labeled[1]])[0]
                    tmp = np.asarray([_zero_one(_) for _ in tmp])
                    print(tmp.reshape((28, 28)))

                    hoge = [0.0] * 10
                    hoge[2] = 1.0
                    tmp = model.generate(sess, z1, [hoge])[0]
                    tmp = np.asarray([_zero_one(_) for _ in tmp])
                    print(tmp.reshape((28, 28)))
                    print("-----")


        print("-- end train--");sys.stdout.flush()
        z1 = model.encode(sess, [batch_figs[1]])[0]
        z2 = model.encode(sess, [batch_figs[8]])[0]
        diff = [z1[i] - z2[i] for i in range(z_dim)]
        
        for i in range(20):
            z = [z2[_] + diff[_] * i * 0.05 for _ in range(z_dim)]
            tmp = model.generate(sess, [z])
            plt.imshow(tmp, cmap = plt.cm.gray)
            #plt.show()
            plt.savefig(os.path.join(dump_dir, "fig{}.png".format(i)))
