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

logging.getLogger().setLevel(logging.INFO)

config = Config()
n_steps = config.step

# placeholders
inputs_ph = tf.placeholder(tf.float32,
                           [None, config.cell_out_size])
locs_ph = tf.placeholder(tf.float32, [None, 2])

w = weight_variable((config.cell_out_size, config.loc_dim))
b = bias_variable((config.loc_dim,))
locs = tf.clip_by_value(tf.nn.xw_plus_b(inputs_ph, w, b), -1., 1.)
loss_op = tf.nn.l2_loss(locs - locs_ph)
train_op = tf.train.AdamOptimizer().minimize(loss_op)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  train_locs = np.load('train_distill.npz')['locs']
  train_inputs = np.load('train_distill.npz')['rnn_output']

  train_locs = train_locs.reshape((55000 * 6, 2))
  train_inputs = train_inputs.reshape((55000 * 6, 256))

  for i in xrange(n_steps):
    batch_inds = np.random.randint(0, high=54999, size=config.batch_size)
    locs = train_locs[batch_inds]
    inputs = train_inputs[batch_inds]
    loss, _ = sess.run(
            [loss_op, train_op],
            feed_dict={
                inputs_ph: inputs,
                locs_ph: locs
            })

    if i and i % 1000 == 0:
      logging.info('step {}: loss = {:3.6f}'.format(i, loss))
