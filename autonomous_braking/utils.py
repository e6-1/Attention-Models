from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np

distributions = tf.contrib.distributions


def get_bucket(num_buckets, x, y, width, height):
    """Get the ``bucket'' that an (x, y) coordinate falls into in an image
    given the number of buckets and width/height."""
    num_buckets = int(sqrt(num_buckets))
    delta_width = width / num_buckets
    delta_height = height / num_buckets
    for i in range(num_buckets):
        if i == (num_buckets - 1):
            for j in range(num_buckets):
                if j == (num_buckets - 1):
                    return num_buckets * j + i
                if y <= (j + 1) * delta_height:
                    return num_buckets * j + i
        if x <= (i + 1) * delta_width:
            for j in range(num_buckets):
                if j == (num_buckets - 1):
                    return num_buckets * j + i
                if y <= (j + 1) * delta_height:
                    return (num_buckets * j + i)
    return -1


def get_sub_seq(seq, start, end):
    """Get the sub sequence starting at the start index and ending at the end index."""
    arr = seq[max([0, start]):end]
    if start < 0:
        arr = np.append(np.zeros((abs(start),2)), arr, axis=0)
    for i in range(len(arr)):
        if np.sum(arr[i]) == 0:
            arr[i] = [1, 0]
    return arr


def minibatch(data, batch_size, data_size):
    """Generates a minibatch from the given data and parameters."""
    randomized = np.random.permutation(data)
    batches = []
    num_batches = 0
    while num_batches * batch_size < data_size:
        new_batch = randomized[num_batches * batch_size:(num_batches + 1) * batch_size]
        batches.append(new_batch)
        num_batches += 1
    return batches


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)


def loglikelihood(mean_arr, sampled_arr, sigma):
  mu = tf.stack(mean_arr)  # mu = [timesteps, batch_sz, loc_dim]
  sampled = tf.stack(sampled_arr)  # same shape as mu
  gaussian = distributions.Normal(mu, sigma)
  logll = gaussian.log_prob(sampled)  # [timesteps, batch_sz, loc_dim]
  logll = tf.reduce_sum(logll, 2)
  logll = tf.transpose(logll)  # [batch_sz, timesteps]
  return logll


def softmax_with_temp(logits, T=1):
  numerator = tf.exp(logits / T)
  denomenator = tf.reduce_sum(tf.exp(logits / T))
  return numerator / denomenator

def draw_ram(image_num, mnist_train, num_glimpses, orig_size, win_size):
    im = mnist_train.images[image_num].reshape((orig_size, orig_size))
    # Create figure and axes
    fig, ax = plt.subplots(1)
    patch = patches.Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='r', facecolor='none')

    # initialization function: plot the background of each frame
    def init():
        ax.imshow(im)
        ax.add_patch(patch)
        return [patch]

    # animation function.  This is called sequentially
    def animate(i):
        coords = transform_coords(mnist_train.locs[image_num][i], conf.original_size, conf.original_size)
        patch.set_width(win_size)
        patch.set_height(win_size)
        patch.set_xy(coords)
        return [patch]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=num_glimpses, blit=True)

    anim.save('ram.mp4', fps=num_glimpses, extra_args=['-vcodec', 'libx264'])
