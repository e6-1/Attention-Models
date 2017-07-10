"""Recurrent Models of Visual Attention V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np

from aux_nets import ContextNet, GlimpseNet, LocNet
from utils import minibatch, weight_variable, bias_variable, loglikelihood, softmax_with_temp
from config import Config

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.rnn as rnn_cell
import tensorflow.contrib.legacy_seq2seq as seq2seq

logging.getLogger().setLevel(logging.INFO)

# rnn_cell = tf.nn.rnn_cell
# seq2seq = tf.nn.seq2seq

config = Config()
n_steps = config.step

loc_mean_arr = []
sampled_loc_arr = []

def get_next_input(output, i):
  global loc
  loc, loc_mean = loc_net(output)
  loc = tf.Print(loc, [loc])
  gl_next = gl(loc)
  loc_mean_arr.append(loc_mean)
  sampled_loc_arr.append(loc)
  return gl_next

# placeholders
images_ph = tf.placeholder(tf.float32,
                             [None,
                             config.img_height,
                             config.img_width,
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
context = cl(images_ph)
# loc = tf.random_uniform((N, 2), minval=-1, maxval=1)
loc = loc_net(context)[0]
init_glimpse = gl(loc)

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
softmax = softmax_with_temp(logits)

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
# training_steps_per_epoch = mnist.train.num_examples // config.batch_size
training_steps_per_epoch = 80
starter_learning_rate = config.lr_start
#  decay per training epoch
learning_rate = tf.train.exponential_decay(
    starter_learning_rate,
    global_step,
    training_steps_per_epoch,
    0.97,
    staircase=True)
learning_rate = tf.maximum(learning_rate, config.lr_min)
opt = tf.train.AdamOptimizer(learning_rate)
# opt = tf.train.AdamOptimizer()
train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

# get locations
locs_op = tf.stack(loc_mean_arr, axis=1)

saver = tf.train.Saver()
# Training code
with tf.Session() as sess:
   sess.run(tf.initialize_all_variables())
   # saver.restore(sess, "ram_model.ckpt")
   # print("Model restored.")
   for i in xrange(n_steps):
     data = np.load('/home/data2/vision6/ethpete/ab_data/batch_{0}.npz'.format(i))

     left_imgs = data['left_imgs']
     right_imgs = data['right_imgs']
     center_imgs = data['center_imgs']
     gazes = data['gazes']
     braking = data['braking']

     img_batches = minibatch(center_imgs, 32, 1000)
     label_batches = minibatch(braking[:, 1], 32, 1000)

     for images, labels in zip(img_batches, label_batches):
       logging.info('fraction braking: {:2.2f}'.format(labels.sum() / labels.shape[0]))
       # duplicate M times, see Eqn (2)
       images = np.tile(images, [config.M, 1, 1, 1])
       labels = np.tile(labels, [config.M])
       loc_net.samping = True
       adv_val, baselines_mse_val, xent_val, logllratio_val, \
           reward_val, loss_val, lr_val, _ = sess.run(
               [advs, baselines_mse, xent, logllratio,
                reward, loss, learning_rate, train_op],
               feed_dict={
                   images_ph: images,
                   labels_ph: labels
               })
       if i and i % 1 == 0:
         logging.info('step {}: lr = {:3.6f}'.format(i, lr_val))
         logging.info(
             'step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
                 i, reward_val, loss_val, xent_val))
         logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
             logllratio_val, baselines_mse_val))

       # if i and i % training_steps_per_epoch == 0:
       #   # Evaluation
       #   for dataset in [mnist.validation, mnist.test]:
       #     steps_per_epoch = dataset.num_examples // config.eval_batch_size
       #     correct_cnt = 0
       #     num_samples = steps_per_epoch * config.batch_size
       #     loc_net.sampling = True
       #     for test_step in xrange(steps_per_epoch):
       #       images, labels, _, _ = dataset.next_batch(config.batch_size)
       #       labels_bak = labels
       #       # Duplicate M times
       #       images = np.tile(images, [config.M, 1])
       #       labels = np.tile(labels, [config.M])
       #       softmax_val = sess.run(softmax,
       #                              feed_dict={
       #                                  images_ph: images,
       #                                  labels_ph: labels
       #                              })
       #       softmax_val = np.reshape(softmax_val,
       #                                [config.M, -1, config.num_classes])
       #       softmax_val = np.mean(softmax_val, 0)
       #       pred_labels_val = np.argmax(softmax_val, 1)
       #       pred_labels_val = pred_labels_val.flatten()
       #       correct_cnt += np.sum(pred_labels_val == labels_bak)
       #     acc = correct_cnt / num_samples
       #     if dataset == mnist.validation:
       #       logging.info('valid accuracy = {}'.format(acc))
       #     else:
       #       logging.info('test accuracy = {}'.format(acc))
       #       iter_accs.append(acc)

   # Save model
   save_path = saver.save(sess, "ram_model.ckpt")
   print("Model saved in file: %s" % save_path)
