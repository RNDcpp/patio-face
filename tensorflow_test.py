#-*- coding:utf-8 -*-
import sys
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from datetime import datetime
import random
import time
import os

import get_input_tensor as input_data
import cnn as mynn

STEP=sys.argv[1]

IMAGE_SIZE=64
NUM_CLASS=2
CH_SIZE=3
BATCH_SIZE=200
TRAIN_FILE=['train.csv']
TEST_FILE=['test.csv']
MAX_STEPS=100000
flags = tf.app.flags
FLAGS = flags.FLAGS
keep_prob=tf.placeholder("float")
v_images, v_labels, filename = input_data.load_data_for_test(
        TEST_FILE,
        BATCH_SIZE,
        image_size=IMAGE_SIZE,
        ch_size=CH_SIZE)

#output=mynn.inference2(images,keep_prob,IMAGE_SIZE,CH_SIZE,NUM_CLASS)
validate=mynn.inference(v_images,keep_prob,IMAGE_SIZE,CH_SIZE,NUM_CLASS)
acc = mynn.accuracy(validate, v_labels)


with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep = 0)
    #sess.run(tf.initialize_all_variables())
    model_path='/output'
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    print("Model restore finished")
    # SummaryWriterでグラフを書く
    tf.train.start_queue_runners(sess)
    acc_res,filename_res,actual_res,expect_res=sess.run([acc,filename,validate,v_labels],feed_dict={keep_prob:1.0})
    print('accuracy',acc_res)
    goods = []
    bads = []
    for idx, (act, exp) in enumerate(zip(actual_res, expect_res)):
        if np.argmax(act) == np.argmax(exp):
            goods.append(filename_res[idx])
        else:
            bads.append(filename_res[idx])
    with open('goods.csv','w') as of:
        for f in goods:
            of.write('%s\n'%(f))
    with open('bads.csv','w') as of:
        for f in bads:
            of.write('%s\n'%(f))
        #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

