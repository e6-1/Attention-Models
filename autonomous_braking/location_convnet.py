'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import numpy as np
from random import shuffle
import tensorflow as tf
from utils import minibatch

# Parameters
learning_rate = 0.0001
training_iters = 200000
batches = 126
batch_size = 32
display_step = 10

# Network Parameters
input_width = 244
input_height = 244
input_channels = 1
n_classes = 4 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, input_height, input_width, input_channels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2, l=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, l, 1], strides=[1, k, l, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # x = tf.reshape(x, shape=[-1, input_height, input_width, input_channels])
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([61*61*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
pred_labels = tf.argmax(pred, 1)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)

saver = tf.train.Saver()
# Training code
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "loc_model.ckpt")
    print("Model restored.")
    """
    for epoch in xrange(20):
        rand_batches = range(110)
        shuffle(rand_batches)
        
        for i in rand_batches:
            data = np.load('/home/data2/vision6/ethpete/gaze_data/batch_{0}.npz'.format(i))

            imgs = data['imgs'][:, :, :, 0]
            imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
            buckets = data['buckets']
            img_batches = minibatch(imgs, 32, 1000)
            label_batches = minibatch(buckets, 32, 1000)
            # print("Processing training batch {0}".format(i))
            for images, labels in zip(img_batches, label_batches):
                sess.run(optimizer, feed_dict={x: images, y: labels, keep_prob: dropout})
            # print("Processed training batch {0}".format(i))
        avg_loss = 0
        avg_acc = 0
        nums = 0        
        for j in range(110, 126):
            data = np.load('/home/data2/vision6/ethpete/gaze_data/batch_{0}.npz'.format(j))

            imgs = data['imgs']
            buckets = data['buckets']
            assert not np.any(np.isnan(imgs))
            img_batches = minibatch(imgs, 32, 1000)
            label_batches = minibatch(buckets, 32, 1000)

            avg_acc = 0
            nums = 0
            for images, labels in zip(img_batches, label_batches):

                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: images,
                                                  y: labels,
                                                  keep_prob: 1.})
                avg_acc += acc
                avg_loss += loss
                nums += 1
        avg_acc /= nums
        avg_loss /= nums
        print("Epoch " + str(epoch) + ", Minibatch Loss= " + \
                "{:.6f}".format(avg_loss) + ", Training Accuracy= " + \
                "{:.5f}".format(avg_acc))
    """
    for k in range(1):
        data = np.load('/home/data2/vision6/ethpete/test_data/batch_{0}.npz'.format(k))
        
        imgs = data['imgs']
        buckets = data['buckets']
        print(buckets)
        prediction = sess.run(pred_labels, feed_dict={x: imgs, y: buckets, keep_prob: 1})
        print(prediction)
    # Save model
    save_path = saver.save(sess, "loc_model.ckpt")
    print("Model saved in file: %s" % save_path)
