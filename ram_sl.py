"""Recurrent Models of Visual Attention V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np

from glimpse import GlimpseNet, LocNet
from utils import weight_variable, bias_variable, loglikelihood
from config import Config

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.rnn as rnn_cell
import tensorflow.contrib.legacy_seq2seq as seq2seq

logging.getLogger().setLevel(logging.INFO)

# rnn_cell = tf.nn.rnn_cell
# seq2seq = tf.nn.seq2seq

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

config = Config()
n_steps = config.step

loc_mean_arr = []
sampled_loc_arr = []


def get_next_input(output, i):
  loc, loc_mean = loc_net(output)
  gl_next = gl(loc)
  loc_mean_arr.append(loc_mean)
  sampled_loc_arr.append(loc)
  return gl_next

# placeholders
images_ph = tf.placeholder(tf.float32,
                           [None, config.original_size * config.original_size *
                            config.num_channels])
locs_ph = tf.placeholder(tf.float32, [None, 6, 2])
labels_ph = tf.placeholder(tf.int64, [None])
logits_ph = tf.placeholder(tf.float32, [None, config.num_classes])

# Build the aux nets.
with tf.variable_scope('glimpse_net'):
  gl = GlimpseNet(config, images_ph)
with tf.variable_scope('loc_net'):
  loc_net = LocNet(config)

# number of examples
N = tf.shape(images_ph)[0]
init_loc = tf.random_uniform((N, 2), minval=-1, maxval=1)
init_glimpse = gl(init_loc)
# Core network.
lstm_cell = rnn_cell.LSTMCell(config.cell_size, state_is_tuple=True)
init_state = lstm_cell.zero_state(N, tf.float32)
inputs = [init_glimpse]
inputs.extend([0] * (config.num_glimpses))
outputs, _ = seq2seq.rnn_decoder(
    inputs, init_state, lstm_cell, loop_function=get_next_input)

# Time independent baselines
with tf.variable_scope('baseline'):
  w_baseline = weight_variable((config.cell_output_size, 1))
  b_baseline = bias_variable((1,))
baselines = []
for t, output in enumerate(outputs[1:]):
  baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
  baseline_t = tf.squeeze(baseline_t)
  baselines.append(baseline_t)
baselines = tf.stack(baselines)  # [timesteps, batch_sz]
baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

# Take the last step only.
output = outputs[-1]
# Build classification network.
with tf.variable_scope('cls'):
  w_logit = weight_variable((config.cell_output_size, config.num_classes))
  b_logit = bias_variable((config.num_classes,))
logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
softmax = tf.nn.softmax(logits)

# distillation loss for logits
logits_loss = tf.nn.l2_loss(logits - logits_ph) / (config.batch_size * config.M)

# cross-entropy.
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
xent = tf.reduce_mean(xent) / (config.batch_size * config.M)
pred_labels = tf.argmax(logits, 1)

# mse
loc_loss = tf.norm(tf.stack(loc_mean_arr, axis=1) - locs_ph)

# combined loss
distill_loss = config.lamda * xent + (1 - config.lamda) * logits_loss + loc_loss

# learning rate
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
training_steps_per_epoch = mnist.train.num_examples // config.batch_size
starter_learning_rate = config.lr_start
# decay per training epoch
learning_rate = tf.train.exponential_decay(
    starter_learning_rate,
    global_step,
    training_steps_per_epoch,
    0.97,
    staircase=True)
learning_rate = tf.maximum(learning_rate, config.lr_min)
opt = tf.train.AdamOptimizer(learning_rate)
train_op = tf.train.AdamOptimizer().minimize(distill_loss)

mnist_train = mnist.train
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  # saver.restore(sess, "ram_model.ckpt")
  # print("Model restored.")
  train_locs = np.load('train_distill.npz')['locs']
  train_logits = np.load('train_distill.npz')['logits']
  mnist_train.locs = train_locs
  mnist_train.logits = train_logits
  for i in xrange(n_steps):
    batch_inds = np.random.randint(0, high=54999, size=config.batch_size)
    # images, labels, locs, logits = mnist_train.next_batch(config.batch_size)
    images, labels, locs, logits = mnist_train.images[batch_inds], mnist_train.labels[batch_inds], train_locs[batch_inds], train_logits[batch_inds]
    # duplicate M times, see Eqn (2)
    images = np.tile(images, [config.M, 1])
    labels = np.tile(labels, [config.M])
    locs = np.tile(locs, [config.M, 1, 1])
    logits = np.tile(logits, [config.M, 1])
    loc_net.sampling = True
    loss, location_loss, logit_loss, softmax_loss, _ = sess.run(
            [distill_loss, loc_loss, logits_loss, xent, train_op],
            feed_dict={
                images_ph: images,
                labels_ph: labels,
                logits_ph: logits,
                locs_ph: locs
            })

    if i and i % 1000 == 0:
      logging.info('step {}: loss = {:3.6f}, loc_loss = {:3.6f}, logits_loss = {:3.6f}, softmax_loss = {:3.6f}'.format(i, loss, location_loss, logit_loss, softmax_loss))
    if i and i % training_steps_per_epoch == 0:
      # Evaluation
      for dataset in [mnist.validation, mnist.test]:
        steps_per_epoch = dataset.num_examples // config.eval_batch_size
        correct_cnt = 0
        num_samples = steps_per_epoch * config.batch_size
        loc_net.sampling = True
        for test_step in xrange(steps_per_epoch):
          images, labels, _, _ = dataset.next_batch(config.batch_size)
          labels_bak = labels
          # Duplicate M times
          images = np.tile(images, [config.M, 1])
          labels = np.tile(labels, [config.M])
          softmax_val = sess.run(softmax,
                                 feed_dict={
                                     images_ph: images,
                                     labels_ph: labels
                                 })
          softmax_val = np.reshape(softmax_val,
                                   [config.M, -1, config.num_classes])
          softmax_val = np.mean(softmax_val, 0)
          pred_labels_val = np.argmax(softmax_val, 1)
          pred_labels_val = pred_labels_val.flatten()
          correct_cnt += np.sum(pred_labels_val == labels_bak)
        acc = correct_cnt / num_samples
        if dataset == mnist.validation:
          logging.info('valid accuracy = {}'.format(acc))
        else:
          logging.info('test accuracy = {}'.format(acc))

  # Save model
  # save_path = saver.save(sess, "ram_model.ckpt")
  # print("Model saved in file: %s" % save_path)
