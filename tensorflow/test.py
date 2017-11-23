import tensorflow as tf
import numpy as np
import cPickle
INPUT_SIZE=3072
HIDDEN_UNIT_SIZE=1024
HIDDEN_UNIT_SIZE2=200
TRAIN_DATA_SIZE=10

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

data=unpickle('cifar-10-batches-py/data_batch_1')
#print len(data)
for k,v in data.items():
    print k
#    print len(v)
#    print v[0:100]
#for k in data['data'][0:10]:
#    print k
#    print len(k)
#print data['data']
#print data['labels']

train_input=data['data'][0:8000]
test_input=data['data'][8000:10000]
train_labels=data['labels'][0:8000]
test_labels=data['labels'][8000:10000]

def inference(input_placeholder):
    with tf.name_scope('hidden1') as scope:
        W1=tf.Variable('W1',tf.truncated_normal([INPUT_SIZE,HIDDEN_UNIT_SIZE],0.0,0.01,dtype=tf.float32))
        b1=tf.Variable('b1',tf.constant(0.1,dtype=tf.float32,shape=[1,HIDDEN_UNIT_SIZE]))
        output1=tf.nn.relu(tf.matmul(input_placeholder,W1)+b1)
    with tf.name_scope as scope:
        W2=tf.Variable('W2',tf.truncated_normal([HIDDEN_UNIT_SIZE,HIDDEN_UNIT_SIZE2],0.0,0.01,dtype=tf.float32))
        b2=tf.Variable('b2',tf.constant(0.1,dtype=tf.float32,shape=[1,HIDDEN_UNIT_SIZE2]))
        output2=tf.nn.relu(tf.matmul(output1,W2)+b2)
    with tf.name_scope as scope:
        W3=tf.Variable('W3',tf.truncated_normal([HIDDEN_UNIT_SIZE2,TRAIN_DATA_SIZE]))
        b2=tf.Variable('b3',tf.constant(0.1,dtype=tf.float32,shape=[1,TRAIN_DATA_SIZE]))
        output=tf.nn.softmax(tf.matmul(output2,W3)+b3)
    return tf.nn.l2_normalize(output,0)

def loss(output, train_placeholder, loss_label_placeholder):
    with tf.name_scope('loss') as scope:
        loss=tf.nn.l2_loss(output - train_placeholder,0)
        tf.scalar_summary(loss_label_placeholder, loss)
    return loss

def training(loss):
    with tf.name_scope('training') as scope:
        train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    return train_step
with tf.Graph().as_default():
    train_placeholder = tf.placeholder('float', [None, TRAIN_DATA_SIZE], name='train_placeholder')
    input_placeholder = tf.placeholder('float', [None, INPUT_SIZE], name='input_placeholder')
    loss_label_placeholder = tf.placeholder('string', name='loss_label_placeholder')
    feed_dict_train={
            train_placeholder: train_labels,
            input_placeholder: train_input,
            loss_label_placeholder: 'loss_train'
            }
    feed_dict_test={
            train_placeholder: test_labels,
            input_placeholder: test_input,
            loss_label_placeholder: 'loss_test'
            }
    output = inference(input_placeholder)
    loss = loss(output, train_placeholder, loss_label_placeholder)
    training_op = training(loss)
    init = tf.initialize_all_variables()
    best_loss=float('inf')
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter('data', graph_def=sess.graph_def)
        sess.run(init)
        for step in range(10000):
            sess.run(training_op,feed_dict_train)
            loss_test=sess.run(loss,feed_dict_test)
            if loss_test<best_loss:
                best_loss = loss_test
                best_mattch = sess.run(output, feed_dict=feed_dict_test)
            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict_test)
                summary_str += sess.run(summary_op, feed_dict=feed_dict_train)
                summary_writer.add_summary(summary_str, step)
        print sess.run(tf.nn.l2_normalize(salary_placeholder, 0), feed_dict=feed_dict_test)
        print best_match
#holder=tf.placeholder(tf.int32,[None])
#const=tf.constant(5)
#add_op=holder+const
#with tf.Session() as sess:
#    result = sess.run(add_op,feed_dict={holder:[5]})
#    print(result)
#    result = sess.run(add_op,feed_dict={holder:[5,15]})
#    print(result)
