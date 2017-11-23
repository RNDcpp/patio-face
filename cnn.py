#-*- coding:utf-8 -*-
import tensorflow as tf
NUM_CLASS=2
def conv2d(value,weight):
    return tf.nn.conv2d(value,weight,strides=[1,1,1,1],padding='SAME')
def max_pool(value,name):
    return tf.nn.max_pool(value,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name=name)
def weight_valiable(name,shape,stddev=5e-2,wd=0.0):
    var=tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def local_norm(value,name):
    return tf.nn.lrn(value,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name=name)

def inference(images_placeholder,keep_prob,image_size,ch_size,num_classes,validate=False):
    x_image = tf.reshape(images_placeholder, [-1, image_size, image_size, ch_size])
    #[image_size*image_size*3ch]を一つの塊としてsample数を1次元めに入れる変換
    #imageを4次元テンソルにする
    print(x_image)
    with tf.variable_scope('conv1') as scope:
        if validate:
            tf.get_variable_scope().reuse_variables()
        #[パッチサイズ(x,y),入力チャネル数,出力チャネル数]
        W_conv1=weight_valiable('W_conv1',[5,5,ch_size,64])
        #バイアスは各チャネルごとに一つ Wx+b
        b_conv1=tf.get_variable('b_conv1',[64],initializer=tf.constant_initializer([0.0]))
        h_conv1=tf.nn.relu(tf.nn.bias_add(conv2d(x_image,W_conv1),b_conv1))
        print(h_conv1)
    with tf.variable_scope('pool1') as scope:
        if validate:
            tf.get_variable_scope().reuse_variables()
        #X,Y方向のサイズはそれぞれ半分になる
        h_pool1=max_pool(h_conv1,'pool1')
        h_norm1=local_norm(h_pool1,'norm1')
        print(h_pool1)
    with tf.variable_scope('conv2') as scope:
        if validate:
            tf.get_variable_scope().reuse_variables()
        W_conv2=weight_valiable('W_conv2',[5,5,64,64])
        b_conv2=tf.get_variable('b_conv2',[64],initializer=tf.constant_initializer([0.1]))
        h_conv2=tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1,W_conv2),b_conv2))
        print(h_conv2)
    with tf.variable_scope('pool2') as scope:
        if validate:
            tf.get_variable_scope().reuse_variables()
        #さらに半分になる
        h_norm2=local_norm(h_conv2,'norm2')
        h_pool2=max_pool(h_norm2,'pool2')
        print(h_pool2)
    with tf.variable_scope('conv3') as scope:
        if validate:
            tf.get_variable_scope().reuse_variables()
        W_conv3=weight_valiable('W_conv3',[5,5,64,64])
        b_conv3=tf.get_variable('b_conv3',[64],initializer=tf.constant_initializer([0.1]))
        h_conv3=tf.nn.relu(tf.nn.bias_add(conv2d(h_pool2,W_conv3),b_conv3))
        print(h_conv3)
    with tf.variable_scope('pool3') as scope:
        if validate:
            tf.get_variable_scope().reuse_variables()
        #さらに半分になる
        h_norm3=local_norm(h_conv3,'norm3')
        h_pool3=max_pool(h_norm3,'pool3')
        print(h_pool3)
    with tf.variable_scope('fc1') as scope:
        if validate:
            tf.get_variable_scope().reuse_variables()
        size_h_pool3=int(image_size/8*image_size/8*64)
        #pool2の空間方向のサイズはimage_sizeのそれぞれ8分の1になっている
        h_pool3_flat=tf.reshape(h_pool3,[-1,size_h_pool3])
        #h_pool2の空間方向の次元をフラットにする[sample,特徴]
        W_fc1=weight_valiable('W_fc1',[size_h_pool3,384],stddev=0.04,wd=0.004)
        b_fc1=tf.get_variable('b_fc1',[384],initializer=tf.constant_initializer([0.1]))
        h_fc1=tf.nn.relu(tf.nn.bias_add(tf.matmul(h_pool3_flat,W_fc1),b_fc1))
        print(h_fc1)
    with tf.variable_scope('fc2') as scope:
        if validate:
            tf.get_variable_scope().reuse_variables()
        h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
        W_fc2=weight_valiable('W_fc2',[384,192],stddev=0.04,wd=0.004)
        b_fc2=tf.get_variable('b_fc2',[192],initializer=tf.constant_initializer([0.1]))
        h_fc2=tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc1_drop,W_fc2),b_fc2))
    with tf.variable_scope('softmax') as scope:
        if validate:
            tf.get_variable_scope().reuse_variables()
        h_fc2_drop=tf.nn.dropout(h_fc2,keep_prob)
        W_fc3=weight_valiable('W_fc3',[192,NUM_CLASS],stddev=1/192.0)
        b_fc3=tf.get_variable('b_fc3',[NUM_CLASS],initializer=tf.constant_initializer([0.0]))
        output=tf.nn.softmax(tf.nn.bias_add(tf.matmul(h_fc2_drop,W_fc3),b_fc3))
    return output

def loss(output,supervisor_labels_placeholder):
    #reduce_meanは平均を計算する
    #reduce_sumは総和を計算する axis=1...2次元めで総和をとる
    #この場合はtf.log(output)が予測されたラベルの列 [[a1,a2],...] 
    #教師データのラベル [[1,0],..] = [[y1,y2],..] から
    #[[y1*log(a1),y2*log(y2)],..]を計算し総和を求める 
    #[sample1の交差エントロピー,sample2の交差エントロピー,...]
    #の一次元テンソルができる
    with tf.name_scope('cross_entropy') as scope:
        cross_entropy=tf.reduce_mean(-tf.reduce_sum(supervisor_labels_placeholder * tf.log(tf.clip_by_value(output,1e-3,1.0)), axis=1))
        tf.summary.scalar("cross_entropy",cross_entropy)
    with tf.name_scope('output_mean') as scope:
        output_mean=tf.reduce_mean(tf.reduce_mean(tf.clip_by_value(output,1e-10,1.0), axis=1))
        tf.summary.scalar("output_mean",output_mean)
    with tf.name_scope('labels_mean') as scope:
        labels_mean=tf.reduce_mean(tf.reduce_mean(supervisor_labels_placeholder, axis=1))
        tf.summary.scalar("labels_mean",labels_mean)
    return cross_entropy

def training(loss):
    #train_step=tf.train.AdamOptimizer(1e-13).minimize(loss)
    train_step=tf.train.AdadeltaOptimizer().minimize(loss)
    #train_step=tf.train.MomentumOptimizer(1e-4,0.5).minimize(loss)
    return train_step

def accuracy(output,labels):
    with tf.name_scope('accuracy') as scope:
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuracy", accuracy)
    return accuracy

