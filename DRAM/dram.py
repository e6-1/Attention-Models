"""Recurrent Models of Visual Attention V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np

from dram_glimpse import GlimpseNet, LocNet, ContextNet
from utils import weight_variable, bias_variable, loglikelihood
from config import Config

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.rnn as rnn_cell
import tensorflow.contrib.legacy_seq2seq as seq2seq
from tensorflow.python.ops.rnn import raw_rnn

# rnn_cell = tf.contrib.rnn
# seq2seq = tf.contrib.legacy_seq2seq

logging.getLogger().setLevel(logging.INFO)

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

config = Config()
n_steps = config.step

loc_mean_arr = []
sampled_loc_arr = []


def get_next_input(output, i):
  global context
  second_lstm_output, context = second_lstm(output, context)
  loc, loc_mean = loc_net(second_lstm_output)
  gl_next = gl(loc)
  loc_mean_arr.append(loc_mean)
  sampled_loc_arr.append(loc)
  return gl_next

# placeholders
images_ph = tf.placeholder(tf.float32,
                           [None,
                           config.original_size,
                           config.original_size,
                           config.num_channels])
labels_ph = tf.placeholder(tf.int64, [None])

# Build the aux nets.
with tf.variable_scope('glimpse_net'):
  gl = GlimpseNet(config, images_ph)
with tf.variable_scope('loc_net'):
  loc_net = LocNet(config)
with tf.variable_scope('context_net'):
  cl = ContextNet(config)

# number of examples
N = tf.shape(images_ph)[0]

# Process context
context = cl(images_ph)
loc = loc_net(context)[0]
glimpse = gl(loc)

# Core network.
with tf.variable_scope('first', reuse=None):
  first_lstm = tf.contrib.rnn.BasicLSTMCell(
            config.cell_size, forget_bias=0.0, state_is_tuple=False,
            reuse=tf.get_variable_scope().reuse)
with tf.variable_scope('second', reuse=None):
  second_lstm = tf.contrib.rnn.BasicLSTMCell(
            config.cell_size, forget_bias=0.0, state_is_tuple=False,
            reuse=tf.get_variable_scope().reuse)
first_state = tf.zeros((N, 2*config.cell_size))
second_state = tf.concat([context, context], axis=1)

outputs = []
for time_step in range(config.num_glimpses):
  with tf.variable_scope('first'):
    if time_step > 0: tf.get_variable_scope().reuse_variables()
    first_output, first_state = first_lstm(glimpse, first_state)
  with tf.variable_scope('second'):
    if time_step > 0: tf.get_variable_scope().reuse_variables()
    second_output, second_state = second_lstm(first_output, second_state)
  loc, loc_mean = loc_net(second_output)
  glimpse = gl(loc)
  loc_mean_arr.append(loc_mean)
  sampled_loc_arr.append(loc)
  outputs.append(first_output)

# inputs = [init_glimpse]
# inputs.extend([0] * (config.num_glimpses))
# outputs, _ = seq2seq.rnn_decoder(
#     decoder_inputs=inputs,
#     initial_state=context,
#     cell=first_lstm,
#     loop_function=get_next_input)

# Time independent baselines
with tf.variable_scope('baseline'):
  w_baseline = weight_variable((config.cell_output_size, 1))
  b_baseline = bias_variable((1,))
baselines = []
for t, output in enumerate(outputs):
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

# cross-entropy.
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
xent = tf.reduce_mean(xent)
pred_labels = tf.argmax(logits, 1)
# 0/1 reward.
reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32)
rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
rewards = tf.tile(rewards, (1, config.num_glimpses))  # [batch_sz, timesteps]
logll = loglikelihood(loc_mean_arr, sampled_loc_arr, config.loc_std)
advs = rewards - tf.stop_gradient(baselines)
logllratio = tf.reduce_mean(logll * advs)
reward = tf.reduce_mean(reward)

baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
var_list = tf.trainable_variables()
# hybrid loss
loss = -logllratio + xent + baselines_mse  # `-` for minimize
grads = tf.gradients(loss, var_list)
grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

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
train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for i in xrange(n_steps):
    images, labels, _, _ = mnist.train.next_batch(config.batch_size)
    images = images.reshape((images.shape[0], config.original_size, config.original_size, config.num_channels))
    # duplicate M times, see Eqn (2)
    images = np.tile(images, [config.M, 1, 1, 1])
    labels = np.tile(labels, [config.M])
    loc_net.sampling = True
    adv_val, baselines_mse_val, xent_val, logllratio_val, \
        reward_val, loss_val, lr_val, _ = sess.run(
            [advs, baselines_mse, xent, logllratio,
             reward, loss, learning_rate, train_op],
            feed_dict={
                images_ph: images,
                labels_ph: labels
            })
    if i and i % 1000 == 0:
      logging.info('step {}: lr = {:3.6f}'.format(i, lr_val))
      logging.info(
          'step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
              i, reward_val, loss_val, xent_val))
      logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
          logllratio_val, baselines_mse_val))

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
          images = images.reshape((images.shape[0], config.original_size, config.original_size, config.num_channels))
          # duplicate M times, see Eqn (2)
          images = np.tile(images, [config.M, 1, 1, 1])
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
