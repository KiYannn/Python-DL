"""
MNIST单层隐藏层模型
包括Dropout和Eager执行的样例
"""
from __future__ import absolute_import, division, print_function


import os

import argparse
import tqdm
import tensorflow as tf
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS=None


def main(_):
    # Step 1: Read in data
    train, test = tf.keras.datasets.mnist.load_data(path=FLAGS.mnist_folder)

    # Step 2: Create datasets and iterator
    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.shuffle(55000)  # if you want to shuffle your data
    train_data = train_data.batch(FLAGS.batch_size)
    train_data = train_data.map(__flatten)


    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(FLAGS.batch_size)
    test_data = test_data.map(__flatten)

    iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                               train_data.output_shapes)
    img, label = iterator.get_next()

    train_init = iterator.make_initializer(train_data)  # initializer for train_data
    test_init = iterator.make_initializer(test_data)  # initializer for test_data

    # Step 4: create weights and bias
    # w is initialized to random variables with mean of 0, stddev of 0.01
    # b is initialized to 0
    # shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
    # shape of b depends on Y
    # 第一层
    w_1 = tf.get_variable(name='weights1', shape=[784, FLAGS.hidden_nodes], initializer=tf.random_normal_initializer())
    b_1 = tf.get_variable(name="bias1", shape=[1, FLAGS.hidden_nodes], initializer=tf.zeros_initializer())
    z_1 = tf.matmul(img, w_1) + b_1
    y_1 = tf.sigmoid(z_1)

    # 第二层
    w_2 = tf.get_variable(name='weights2', shape=[FLAGS.hidden_nodes, 10], initializer=tf.random_normal_initializer())
    b_2 =tf.get_variable(name="bias2", shape=[1, 10], initializer=tf.zeros_initializer())
    z_2 = tf.matmul(y_1, w_2) + b_2


    # Step 5: define loss function
    # use cross entropy of softmax of logits as the loss function
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=z_2, labels=label, name='loss')
    loss = tf.reduce_mean(entropy)  # computes the mean over all the examples in the batch

    # Step 6: define training op
    # using gradient descent with learning rate of 0.01 to minimize loss
    optimizer = tf.train.AdamOptimizer(FLAGS.eta).minimize(loss)

    # Step 7: calculate accuracy with test set
    preds = tf.nn.softmax(z_2)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    streaming_accuracy, streaming_accuracy_update = tf.contrib.metrics.streaming_mean(correct_preds)

    # 第一层
    with tf.name_scope('layer1'):
        variable_summaries(w_1, 'w_1')
        variable_summaries(b_1, 'b_1')

    # 第二层
    with tf.name_scope('layer2'):
        variable_summaries(w_2, 'w_2')
        variable_summaries(b_2, 'b_2')

    with tf.name_scope('Accuracy'):
        tf.summary.scalar('accuracy', accuracy)
        streaming_accuracy_scalar = tf.summary.scalar('accuracy', streaming_accuracy)

    with tf.name_scope('Loss'):
        loss_scalar = tf.summary.scalar('Loss', loss)

    # 把所有的summary合到一张图上
    train_merged = tf.summary.merge_all()
    test_merged = tf.summary.merge([streaming_accuracy_scalar])


    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

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

def __flatten(x,y):
    x = tf.to_float(tf.reshape(x,[FLAGS.batch_size,784]))
    norm = tf.realdiv(x,255.0)
    y = tf.one_hot(y, 10)
    return  norm, y

def variable_summaries(var, name):
    with tf.name_scope('summary_'+name):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)  #输出平均值
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev) #输出标准差
        tf.summary.scalar('max',tf.reduce_max(var)) #输出最大值
        tf.summary.scalar('min',tf.reduce_min(var)) #输出最小值
        tf.summary.histogram('histogram',var) #输出柱状图


if __name__ == '__main__':
    # Define paramaters for the model
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist_folder", default='data') #mnist数据下载目录
    parser.add_argument("--eta",default=0.01,type=float) #学习率
    parser.add_argument("--epochs",default=10,type=int) #训练次数
    parser.add_argument("--batch_size",default=1000,type=int) #每次训练批量
    parser.add_argument("--test_interval",default=10, type=int) #测试间隔
    parser.add_argument("--hidden_nodes", default=30, type=int)  #隐藏层节点
    parser.add_argument("--log_dir",default='log/')  #日志目录
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
