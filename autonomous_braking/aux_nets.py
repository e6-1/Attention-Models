from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import weight_variable, bias_variable, conv2d


class GlimpseNet(object):
  """Glimpse network.

  Take glimpse location input and output features for RNN.

  """

  def __init__(self, config, images_ph):
    self.original_size = config.original_size
    self.num_channels = config.num_channels
    self.sensor_size = config.sensor_size
    self.win_size = config.win_size
    self.minRadius = config.minRadius
    self.depth = config.depth

    self.hg_size = config.hg_size
    self.hl_size = config.hl_size
    self.g_size = config.g_size
    self.loc_dim = config.loc_dim

    self.images_ph = images_ph

    self.init_weights()

  def init_weights(self):
    """ Initialize all the trainable weights."""
    self.w_g0 = weight_variable((self.sensor_size, self.hg_size))
    self.b_g0 = bias_variable((self.hg_size,))
    self.w_l0 = weight_variable((self.loc_dim, self.hl_size))
    self.b_l0 = bias_variable((self.hl_size,))
    self.w_g1 = weight_variable((self.hg_size, self.g_size))
    self.b_g1 = bias_variable((self.g_size,))
    self.w_l1 = weight_variable((self.hl_size, self.g_size))
    self.b_l1 = weight_variable((self.g_size,))

  def get_glimpse(self, loc):
    """Take glimpse on the original images."""
    glimpse_imgs = tf.image.extract_glimpse(input=self.images_ph,
                                            size=[self.win_size, self.win_size], offsets=loc)
    # glimpse_imgs = tf.reshape(glimpse_imgs, [
    #     tf.shape(loc)[0], self.win_size * self.win_size * self.num_channels
    # ])
    return glimpse_imgs

  def __call__(self, loc):
    glimpse_input = self.get_glimpse(loc)
    # glimpse_input = tf.reshape(glimpse_input,
    #                            (tf.shape(loc)[0], self.sensor_size))
    # g = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, self.w_g0, self.b_g0))
    # g = tf.nn.xw_plus_b(g, self.w_g1, self.b_g1)
    # l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
    # l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)
    # g = tf.nn.relu(g + l)
    # return g

    # Create Glimpse network
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([32, 32, 3, 32])),
        # 5x5 conv, 32 inputs, 32 outputs
        'wc2': tf.Variable(tf.random_normal([32, 32, 32, 32])),
        # 5x5 conv, 32 inputs, 32 outputs
        'wc3': tf.Variable(tf.random_normal([32, 32, 32, 32])),
        # fully connected, 8*8*32 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([32*32*32, self.g_size])),
        # fully connected, 2 inputs, 1024 outputs
        'wd2': tf.Variable(tf.random_normal([2, self.g_size]))

    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([32])),
        'bc3': tf.Variable(tf.random_normal([32])),
        'bd1': tf.Variable(tf.random_normal([self.g_size])),
        'bd2': tf.Variable(tf.random_normal([self.g_size]))
    }

    # Image network
    conv1 = conv2d(glimpse_input, weights['wc1'], biases['bc1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    fcg = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fcg = tf.add(tf.matmul(fcg, weights['wd1']), biases['bd1'])
    # Add ReLu here?

    # Location network
    fcloc = tf.add(tf.matmul(loc, weights['wd2']), biases['bd2'])
    # Add ReLu here?

    # Combine location with image ("What and Where")
    out = tf.multiply(fcg, fcloc)
    return out


class LocNet(object):
  """Location network.

  Take output from other network and produce and sample the next location.

  """

  def __init__(self, config):
    self.loc_dim = config.loc_dim
    self.input_dim = config.cell_output_size
    self.loc_std = config.loc_std
    self._sampling = True

    self.init_weights()

  def init_weights(self):
    self.w = weight_variable((self.input_dim, self.loc_dim))
    self.b = bias_variable((self.loc_dim,))

  def __call__(self, input):
    mean = tf.clip_by_value(tf.nn.xw_plus_b(input, self.w, self.b), -1., 1.)
    mean = tf.stop_gradient(mean)
    if self._sampling:
      loc = mean + tf.random_normal(
          (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
      loc = tf.clip_by_value(loc, -1., 1.)
    else:
      loc = mean
    loc = tf.stop_gradient(loc)
    return loc, mean

  @property
  def sampling(self):
    return self._sampling

  @sampling.setter
  def sampling(self, sampling):
    self._sampling = sampling


class ContextNet(object):
  """Context network.

  Takes downsampled input image and creates feature vector for initial hidden state of core network.

  """

  def __init__(self, config):
    self.weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([32, 32, 3, 32])),
        # 5x5 conv, 32 inputs, 32 outputs
        'wc2': tf.Variable(tf.random_normal([32, 32, 32, 32])),
        # 5x5 conv, 32 inputs, 32 outputs
        'wc3': tf.Variable(tf.random_normal([32, 32, 32, 32])),
        # fully connected, 16*16*32 inputs, 256 outputs
        'wd1': tf.Variable(tf.random_normal([32*32*32, 256]))
    }
    self.biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([32])),
        'bc3': tf.Variable(tf.random_normal([32])),
        'bd1': tf.Variable(tf.random_normal([256]))
    }
    self.resize_shape = tf.constant([32, 32])

  def __call__(self, imgs):
    imgs = tf.image.resize_images(imgs, self.resize_shape)

    # Image network
    conv1 = conv2d(imgs, self.weights['wc1'], self.biases['bc1'])
    conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
    conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'])
    fc = tf.reshape(conv3, [-1, self.weights['wd1'].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, self.weights['wd1']), self.biases['bd1'])
    return fc
