"""Recurrent Models of Visual Attention V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np

from glimpse_sl import GlimpseNet, LocNet
from utils import weight_variable, bias_variable, loglikelihood
from config import Config

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.rnn as rnn_cell
import tensorflow.contrib.legacy_seq2seq as seq2seq

logging.getLogger().setLevel(logging.INFO)

# rnn_cell = tf.nn.rnn_cell
# seq2seq = tf.nn.seq2seq

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
mnist_train = mnist.train

config = Config()
n_steps = config.step

loc_mean_arr = []
sampled_loc_arr = []

TEMPERATURE = 20

def get_next_input(output, i):
  loc, loc_mean = loc_net(output)
  loc = tf.where(is_distilling, locs_list[i - 1], loc)
  gl_next = gl(loc)  # model's location
  loc_mean_arr.append(loc_mean)
  sampled_loc_arr.append(loc)
  return gl_next

# placeholders
images_ph = tf.placeholder(tf.float32,
                           [None, config.original_size * config.original_size *
                            config.num_channels])
labels_ph = tf.placeholder(tf.int64, [None])
locs_ph = tf.placeholder(tf.float32, [None, 6, 2])
labels_ph = tf.placeholder(tf.int64, [None])
targets_ph = tf.placeholder(tf.float32, [None, config.num_classes])
is_distilling = tf.placeholder(tf.bool)
targets_w_temp = tf.div(targets_ph, TEMPERATURE)
zero_targets_ph = targets_ph - tf.reduce_mean(targets_ph)  # zero mean
locs_list = tf.split(value=locs_ph, num_or_size_splits=6, axis=1)
locs_list = [tf.squeeze(l) for l in locs_list]

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
logits_w_temp = tf.div(logits, TEMPERATURE)
zero_logits = logits - tf.reduce_mean(logits)  # zero mean
softmax = tf.nn.softmax(logits)

# student-teacher loss
zero = tf.constant(0.0)
# distill_lambda = tf.Variable(0.5, name="distill_lambda")
distill_lambda = tf.constant(0.5, name="distill_lambda")
distill_scaling = tf.maximum(distill_lambda, zero)
ndistill_scaling = tf.maximum(1 - distill_lambda, zero)
distill_logits = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_w_temp, labels=targets_w_temp))
distill_locs = tf.nn.l2_loss(tf.stack(loc_mean_arr, axis=1) - locs_ph)
distill_loss = distill_scaling * distill_logits + distill_locs
# distill_loss = distill_scaling * tf.reduce_sum(tf.scalar_mul(0.5, tf.square(zero_logits - zero_targets_ph))) + tf.nn.l2_loss(tf.stack(loc_mean_arr, axis=1) - locs_ph)

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
loss = -logllratio + ndistill_scaling * xent + baselines_mse + distill_loss  # `-` for minimize
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

# get locations
locs_op = tf.stack(loc_mean_arr, axis=1)

saver = tf.train.Saver()

# Training code
with tf.Session() as sess:
   sess.run(tf.initialize_all_variables())
   # saver.restore(sess, "ram_model.ckpt")
   # print("Model restored.")
   train_locs = np.load('train_distill.npz')['locs']
   train_logits = np.load('train_distill.npz')['logits']
   mnist_train.locs = train_locs
   mnist_train.logits = train_logits
   dummy_locs = np.zeros((config.batch_size * config.M, 6, 2))
   dummy_logs = np.zeros((config.batch_size * config.M, config.num_classes))
   for i in xrange(n_steps):
     images, labels, locs, logs = mnist_train.next_batch(config.batch_size)
     # duplicate M times, see Eqn (2)
     images = np.tile(images, [config.M, 1])
     labels = np.tile(labels, [config.M])
     locs = np.tile(locs, [config.M, 1, 1])
     logs = np.tile(logs, [config.M, 1])
     adv_val, baselines_mse_val, xent_val, logllratio_val, \
         reward_val, loss_val, lr_val, d, _ = sess.run(
             [advs, baselines_mse, xent, logllratio,
              reward, loss, learning_rate, distill_logits, train_op],
             feed_dict={
                 images_ph: images,
                 labels_ph: labels,
                 locs_ph: locs,
                 targets_ph: logs,
                 is_distilling: True
             })
     if i and i % 1000 == 0:
       logging.info('step {}: lr = {:3.6f}'.format(i, lr_val))
       logging.info(
           'step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
               i, reward_val, loss_val, xent_val))
       logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
           logllratio_val, baselines_mse_val))
       logging.info('distill_lambda = {:3.4f}'.format(sess.run(distill_lambda)))
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
                                      labels_ph: labels,
                                      locs_ph: dummy_locs,
                                      targets_ph: dummy_logs,
                                      is_distilling: False
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
   save_path = saver.save(sess, "ram_model.ckpt")
   print("Model saved in file: %s" % save_path)

