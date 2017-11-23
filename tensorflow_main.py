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

#LOGDIR='tmp/data.%s' % datetime.now().isoformat()
LOGDIR='/output'
#os.mkdir(LOGDIR)
print(LOGDIR)

DATADIR='/mydata'

IMAGE_SIZE=64
NUM_CLASS=2
#NUM_CLASS=10
CH_SIZE=3
BATCH_SIZE=128
TRAIN_FILE=['train.csv']
#TRAIN_FILE=[os.path.join(DATADIR, 'data_batch_%d.bin' % i)for i in range(1, 6)]
#TEST_FILE=[os.path.join(DATADIR, 'data_batch_%d.bin' % i)for i in range(1, 6)]
TEST_FILE=['test.csv']
MAX_STEPS=100000
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train', 'train.csv', 'File name of train data')
flags.DEFINE_string('test', 'test.csv', 'File name of train data')
flags.DEFINE_string('train_dir', LOGDIR, 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', MAX_STEPS, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

keep_prob=tf.placeholder("float")
#"""
images, labels, _ = input_data.load_data(
        TRAIN_FILE,
        BATCH_SIZE,
        image_size=IMAGE_SIZE,
        ch_size=CH_SIZE,
        shuffle = True,
        distored = True)
v_images, v_labels, _ = input_data.load_data(
        TRAIN_FILE,
        BATCH_SIZE,
        image_size=IMAGE_SIZE,
        ch_size=CH_SIZE,
        shuffle = True,
        distored = True)
"""
images, labels, _ = input_data.load_cifar10(
        TRAIN_FILE,
        BATCH_SIZE,
        image_size=IMAGE_SIZE,
        ch_size=CH_SIZE,
        shuffle = True,
        distored = True)
v_images, v_labels, _ = input_data.load_cifar10(
        TRAIN_FILE,
        BATCH_SIZE,
        image_size=IMAGE_SIZE,
        ch_size=CH_SIZE,
        shuffle = True,
        distored = True)
"""

#output=mynn.inference2(images,keep_prob,IMAGE_SIZE,CH_SIZE,NUM_CLASS)
output=mynn.inference(images,keep_prob,IMAGE_SIZE,CH_SIZE,NUM_CLASS)
validate=mynn.inference(v_images,keep_prob,IMAGE_SIZE,CH_SIZE,NUM_CLASS,validate=True)
loss=mynn.loss(output,labels)
train_op=mynn.training(loss)
acc = mynn.accuracy(validate, v_labels)


with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep = 0)
    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(sess,'/output/')
    print(ckpt)
    saver.restore(sess, '/output/model.ckpt-%s'%(94000))
    # SummaryWriterでグラフを書く
    tf.train.start_queue_runners(sess)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(LOGDIR, graph=sess.graph)
    for step in range(MAX_STEPS):
        start_time = time.time()
        _, loss_result = sess.run([train_op, loss], feed_dict={keep_prob: 0.98})
        acc_res = sess.run([acc], feed_dict={keep_prob: 1.00})
        duration = time.time() - start_time
        if step % 10 == 0:
            num_examples_per_step = BATCH_SIZE
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print(format_str % (datetime.now(), step, loss_result, examples_per_sec, sec_per_batch))
            print('acc_res', acc_res)
        if step % 100 == 0:
            summary_str = sess.run(summary_op,feed_dict={keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)
        if step % 1000 == 0 or (step + 1) == MAX_STEPS or acc_res == 1.0:
            checkpoint_path = '/output/model.ckpt'
            saver.save(sess, checkpoint_path, global_step=step)
        if acc_res == 1.0:
            print('loss is zero')
            break

