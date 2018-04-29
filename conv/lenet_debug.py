"""
MNIST LeNet模型
TensorFlow v 1.8
"""
from __future__ import absolute_import, division, print_function


import os

import argparse
import tensorflow as tf
import sys
from tensorflow.python import debug as tf_debug

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS=None


def main(_):
    # Step 1: Read in data, relative to ~/.keras/datasets
    train, test = tf.keras.datasets.mnist.load_data(path=FLAGS.mnist_folder)

    # Step 2: Create datasets and iterator
    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.shuffle(55000)  # if you want to shuffle your data
    train_data = train_data.batch(FLAGS.batch_size)
    train_data = train_data.map(__normalize)


    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(FLAGS.batch_size)
    test_data = test_data.map(__normalize)

    iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                               train_data.output_shapes)
    img, label = iterator.get_next()

    train_init = iterator.make_initializer(train_data)  # initializer for train_data
    test_init = iterator.make_initializer(test_data)  # initializer for test_data

    # Expand dimension
    img = tf.expand_dims(img, -1)
    # SOLUTION: Layer 1: Convolutional. Input = 28x28x1. with padding 2x2 Output = 28x28x6.
    with tf.name_scope('Conv_1'):
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = 0, stddev = 0.1), name="Conv1_W")
        conv1_b = tf.Variable(tf.zeros(6), name="Conv1_b")
        conv1   = tf.nn.conv2d(img, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

        # SOLUTION: Activation.
        conv1 = tf.nn.relu(conv1)

        # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        #  Image summaries
        conv1_sample = tf.transpose(conv1[0,:,:,:],perm=[2,0,1])
        conv1_sample = tf.expand_dims(conv1_sample, axis = 3)
        conv1_image = tf.summary.image("conv1", conv1_sample, max_outputs= 6)

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    with tf.name_scope('Conv_2'):
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean= 0, stddev=0.1),  name="Conv2_W")
        conv2_b = tf.Variable(tf.zeros(16), name="Conv2_b")
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        # SOLUTION: Activation.
        conv2 = tf.nn.relu(conv2)

        # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        #  Image summaries
        conv2_sample = tf.transpose(conv2[0,:,:,:],perm=[2,0,1])
        conv2_sample = tf.expand_dims(conv2_sample, axis=3)
        conv2_image = tf.summary.image("conv2", conv2_sample, max_outputs= 16)

    # SOLUTION: Layer 3: Input = 5x5x16. Output = 120.
    with tf.name_scope('Conv_3'):
        conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 120), mean= 0, stddev=0.1), name="Conv3_W")
        conv3_b = tf.Variable(tf.zeros(120), name="Conv3_b")
        conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b

        # SOLUTION: Activation.
        conv3 = tf.nn.relu(conv3)

        #  Image summaries
        conv3_sample = tf.transpose(conv3[0,:,:,:],perm=[2,0,1])
        conv3_sample = tf.expand_dims(conv3_sample, axis=3)
        conv3_image = tf.summary.image("conv3", conv3_sample, max_outputs= 120)

    # Removes dimensions of size 1 from the shape of a tensor.
    fc0=tf.squeeze(conv3)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    with tf.name_scope('fc_1'):
        fc1_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=0, stddev=0.1), name="fc1_W")
        fc1_b = tf.Variable(tf.zeros(84),  name="fc1_b")
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        # SOLUTION: Activation.
        fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    with tf.name_scope('fc_2'):
        fc2_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=0, stddev=0.1), name="fc2_W")
        fc2_b = tf.Variable(tf.zeros(10), name="fc2_b")
        z_2 = tf.matmul(fc1, fc2_W) + fc2_b


    # Step 5: define loss function
    # use cross entropy of softmax of logits as the loss function
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z_2, labels=label, name='loss')
    loss = tf.reduce_mean(entropy)  # computes the mean over all the examples in the batch

    # Step 6: define training op
    # using gradient descent with learning rate of 0.01 to minimize loss
    optimizer = tf.train.AdamOptimizer(FLAGS.eta).minimize(loss)

    # Step 7: calculate accuracy with test set
    preds = tf.nn.softmax(z_2)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    streaming_accuracy, streaming_accuracy_update = tf.metrics.mean(correct_preds)


    with tf.name_scope('Accuracy'):
        train_accuracy_scalar = tf.summary.scalar('Train_Accuracy', accuracy)
        streaming_accuracy_scalar = tf.summary.scalar('Test_Accuracy', streaming_accuracy)


    # 把所有的summary合到一张图上
    train_merged = tf.summary.merge([train_accuracy_scalar])
    test_merged = tf.summary.merge([streaming_accuracy_scalar, conv1_image, conv2_image, conv3_image])


    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if FLAGS.debug == 'cli':
            sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type=FLAGS.ui_type)
        else:
            sess = tf_debug.TensorBoardDebugWrapperSession(
                sess, FLAGS.tensorboard_debug_address)

        train_step = 0
        for _ in range(FLAGS.epochs):
            sess.run(train_init)
            try:
                while True:


                    _, l, summaries = sess.run([optimizer, loss, train_merged])
                    train_writer.add_summary(summaries, global_step=train_step)
                    train_step += 1
            except tf.errors.OutOfRangeError:
                pass

            sess.run(test_init)
            try:
                while True:
                    _, summaries = sess.run([streaming_accuracy_update, test_merged])
                    test_writer.add_summary(summaries, global_step=train_step)
            except tf.errors.OutOfRangeError:
                pass

        test_writer.close()
        train_writer.close()

def __normalize(x,y):
    #  computes (x - mean) / adjusted_stddev
    norm = tf.image.per_image_standardization(x)
    y = tf.one_hot(y, 10)
    return  norm, y


if __name__ == '__main__':
    # Define paramaters for the model
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist_folder", default='data') #mnist数据下载目录
    parser.add_argument("--eta",default=0.01,type=float) #学习率
    parser.add_argument("--epochs",default=10,type=int) #训练次数
    parser.add_argument("--batch_size",default=1000,type=int) #每次训练批量
    parser.add_argument("--test_interval",default=10, type=int) #测试间隔
    parser.add_argument("--log_dir",default='log/')  #日志目录
    parser.add_argument(
        "--ui_type",
        type=str,
        default="curses",
        help="Command-line user interface type (curses | readline)")
    parser.add_argument(
        "--debug",
        default="cli",
        help="Use debugger to track down bad values during training ( cli | tensorboard). ")
    parser.add_argument(
        "--tensorboard_debug_address",
        type=str,
        default=None,
        help="Connect to the TensorBoard Debugger Plugin backend specified by "
             "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
             "--debug flag.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
