import tensorflow as tf
import random
NUM_CLASS=2

def load_data_for_test(csv, batch_size,image_size,ch_size):
    return load_data(csv, batch_size, image_size, ch_size, shuffle = False, distored = False)

class Cifar10Record(object):
    def __init__(self):
        self.label_bytes=1
        self.width = 32
        self.height = 32
        self.depth = 3
        self.image_bytes=self.width*self.height*self.depth
        self.record_bytes=self.image_bytes+self.label_bytes

    def load(self, filename_queue):
        reader = tf.FixedLengthRecordReader(record_bytes=self.record_bytes)
        self.key, value = reader.read(filename_queue)
        self.raw = tf.decode_raw(value, tf.uint8)
        self.label = tf.cast(tf.strided_slice(self.raw, [0], [self.label_bytes]), tf.int32)
        self.label.set_shape([1])
        self.label = tf.one_hot(self.label, depth = 10, on_value = 1.0, off_value = 0.0, axis = -1)
        depth_major = tf.reshape(
                tf.strided_slice(self.raw, 
                    [self.label_bytes],
                    [self.record_bytes]
                ),
                [self.depth, self.height, self.width])
        self.uint8image = tf.transpose(depth_major, [1, 2, 0])

def load_cifar10(filenames,batch_size,image_size,ch_size,shuffle=True,distored=True):
    filename_queue=tf.train.string_input_producer(filenames)
    result=Cifar10Record()
    result.load(filename_queue)
    img=result.uint8image
    if distored:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.8)
        img = tf.image.random_hue(img, max_delta=0.4)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.0)
    img=tf.image.resize_images(img,[image_size,image_size])
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(batch_size * min_fraction_of_examples_in_queue)
    return _generate_image_and_label_batch(
            img,
            result.label,
            result.key,
            min_queue_examples, batch_size,
            shuffle=shuffle)

def load_image(filepath,image_size,ch_size):
    img=tf.read_file(filepath)
    img=tf.image.decode_jpeg(img,channels=ch_size)
    img=tf.cast(img,tf.float32)
    img.set_shape([image_size,image_size,ch_size])
    #cropsize = image_size/4*3
    #img = tf.image.resize_image_with_crop_or_pad(img,cropsize,cropsize)
    img=tf.image.resize_images(img,[image_size,image_size])
    #img=tf.image.per_image_standardization(img)
    # Ensure that the random shuffling has good mixing properties.
    #img.set_shape([1,image_size,image_size,ch_size])
    return img

def load_data(csv,batch_size,image_size,ch_size,shuffle=True,distored=True):
    fname_queue = tf.train.string_input_producer(csv, shuffle=shuffle)
    reader = tf.TextLineReader()
    key, val = reader.read(fname_queue)
    fname,label = tf.decode_csv(val, [["path"], [1]])
    label=tf.cast(label,tf.int64)
    label = tf.one_hot(label, depth = NUM_CLASS, on_value = 1.0, off_value = 0.0, axis = -1)
    img=tf.read_file('/mydata/'+fname)
    img=tf.image.decode_jpeg(img,channels=ch_size)
    img=tf.cast(img,tf.float32)
    img.set_shape([image_size,image_size,ch_size])
    #cropsize = random.randint(image_size/2, image_size)
    #img = tf.image.resize_image_with_crop_or_pad(img,cropsize,cropsize)
    if distored:
        #cropsize = random.randint(image_size/2, image_size)
        #img = tf.image.resize_image_with_crop_or_pad(img,cropsize,cropsize)
        img = tf.image.random_flip_left_right(img)
        #img = tf.image.random_brightness(img, max_delta=0.8)
        img = tf.image.random_hue(img, max_delta=0.4)
        #img = tf.image.random_contrast(img, lower=0.8, upper=1.0)
    img=tf.image.resize_images(img,[image_size,image_size])
    #img=tf.image.per_image_standardization(img)
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(batch_size * min_fraction_of_examples_in_queue)
    return _generate_image_and_label_batch(
            img,
            label,
            fname,
            min_queue_examples, batch_size,
            shuffle=shuffle)

def _generate_image_and_label_batch(image, label, filename, min_queue_examples,
                                    batch_size, shuffle):

    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    capacity = min_queue_examples + 3 * batch_size

    if shuffle:
        images, label_batch, filename = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch, filename = tf.train.batch(
            [image, label,filename],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity)

    # Display the training images in the visualizer.
    tf.summary.image('image', images, max_outputs = 100)

    labels = tf.reshape(label_batch, [batch_size, NUM_CLASS])
    return images, labels, filename
