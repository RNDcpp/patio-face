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

TARGET=sys.argv[1]
LNAME=['non-patio','patio']
IMAGE_SIZE=64
NUM_CLASS=2
CH_SIZE=3
BATCH_SIZE=600
TRAIN_FILE=['train.csv']
MAX_STEPS=100000
flags = tf.app.flags
FLAGS = flags.FLAGS
keep_prob=tf.placeholder("float")
image=input_data.load_image(TARGET,image_size=IMAGE_SIZE,ch_size=CH_SIZE)

#output=mynn.inference2(images,keep_prob,IMAGE_SIZE,CH_SIZE,NUM_CLASS)
output=mynn.inference(image,keep_prob,IMAGE_SIZE,CH_SIZE,NUM_CLASS)

with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep = 0)
    #sess.run(tf.initialize_all_variables())
    model_path='/output'
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    print("Model restore finished")
    # SummaryWriterでグラフを書く
    tf.train.start_queue_runners(sess)
    actual_res=sess.run([output],feed_dict={keep_prob:1.0})
    print('result',actual_res)
    print('label',np.argmax(actual_res))
    print('label-name',LNAME[np.argmax(actual_res)])
    print('patio value',actual_res[0][0][1])

