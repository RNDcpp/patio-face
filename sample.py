import tensorflow as tf
x=tf.constant(1,name="X")
y=tf.constant(2,name="Y")
add_op=tf.add(x,y)
with tf.Session() as sess:
    print(sess.run(add_op))
