import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops


def linear2(x, output_dim, hidden_layer_size=64):
  """
  affine transformation Wx+b
  assumes x.shape = (batch_size, num_features)
  """
  #     w = tf.get_variable("w", [x.get_shape()[1], output_dim])  # , initializer=tf.random_normal_initializer()) 
  #   #   b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
  #     b = tf.get_variable("b", [output_dim], initializer=tf.random_normal_initializer())
  #     return tf.matmul(x, w) + b
  
  if hidden_layer_size > 0:
    hidden = tf.contrib.layers.fully_connected(x, hidden_layer_size, activation_fn=tf.nn.relu)
    return tf.contrib.layers.fully_connected(hidden, output_dim, activation_fn=None)
  else:
    return tf.contrib.layers.fully_connected(x, output_dim, activation_fn=None)
  
# def linear(input_, output_size, stddev=0.02, bias_start=0.0, with_w=False, initialization=None, weightnorm=None, gain=1.):
#   shape = input_.get_shape().as_list()
#   input_dim = shape[1]
#   
#   _weights_stdev = None
# 
#   def uniform(stdev, size):
#       if _weights_stdev is not None:
#           stdev = _weights_stdev
#       return np.random.uniform(
#           low=-stdev * np.sqrt(3),
#           high=stdev * np.sqrt(3),
#           size=size
#       ).astype('float32')
# 
#   if initialization == 'lecun':  # and input_dim != output_dim):
#       # disabling orth. init for now because it's too slow
#       weight_values = uniform(
#           np.sqrt(1. / input_dim),
#           (input_dim, output_size)
#       )
# 
#   elif initialization == 'glorot' or (initialization == None):
# 
#       weight_values = uniform(
#           np.sqrt(2. / (input_dim + output_size)),
#           (input_dim, output_size)
#       )
# 
#   elif initialization == 'he':
# 
#       weight_values = uniform(
#           np.sqrt(2. / input_dim),
#           (input_dim, output_size)
#       )
# 
#   elif initialization == 'glorot_he':
# 
#       weight_values = uniform(
#           np.sqrt(4. / (input_dim + output_size)),
#           (input_dim, output_size)
#       )
# 
#   elif initialization == 'orthogonal' or \
#       (initialization == None and input_dim == output_size):
#       
#       # From lasagne
#       def sample(shape):
#           if len(shape) < 2:
#               raise RuntimeError("Only shapes of length 2 or more are "
#                                  "supported.")
#           flat_shape = (shape[0], np.prod(shape[1:]))
#            # TODO: why normal and not uniform?
#           a = np.random.normal(0.0, 1.0, flat_shape)
#           u, _, v = np.linalg.svd(a, full_matrices=False)
#           # pick the one with the correct shape
#           q = u if u.shape == flat_shape else v
#           q = q.reshape(shape)
#           return q.astype('float32')
# 
#       weight_values = sample((input_dim, output_size))
#   
#   elif initialization[0] == 'uniform':
#   
#       weight_values = np.random.uniform(
#           low=-initialization[1],
#           high=initialization[1],
#           size=(input_dim, output_size)
#       ).astype('float32')
# 
#   else:
# 
#       raise Exception('Invalid initialization!')
# 
#   weight_values *= gain
# 
#   try:
#     matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
#                tf.random_normal_initializer(stddev=stddev))
#   except ValueError as err:
#       msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
#       err.args = err.args + (msg,)
#       raise
#     
# #   if weightnorm == None:
# #       weightnorm = _default_weightnorm
# #   if weightnorm:
# #       norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
# #       # norm_values = np.linalg.norm(weight_values, axis=0)
# # 
# #       target_norms = lib.param(
# #           name + '.g',
# #           norm_values
# #       )
# # 
# #       with tf.name_scope('weightnorm') as scope:
# #           norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
# #           weight = weight * (target_norms / norms)
#                 
#   bias = tf.get_variable("bias", [output_size],
#     initializer=tf.constant_initializer(bias_start))
#   if with_w:
#     return tf.matmul(input_, matrix) + bias, matrix, bias
#   else:
#     return tf.matmul(input_, matrix) + bias


if "concat_v2" in dir(tf):

  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)

else:

  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)


class batch_norm(object):

  def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)


def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv

     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak * x)

