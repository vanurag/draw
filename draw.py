#!/usr/bin/env python

""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow

Example Usage: 
	python draw.py --data_dir=/tmp/draw --read_attn=True --write_attn=True

Author: Eric Jang
"""

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import sys
import math
import time
from config import config

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn", True, "enable attention for writer")
FLAGS = tf.flags.FLAGS
A = config['A']
B = config['B']
T = config['T']

# # MODEL PARAMETERS ##

# # MNIST
# A, B = 28, 28  # image width,height
# img_size = B * A  # the canvas size
# enc_size = 256  # number of hidden units / output size in LSTM
# dec_size = 256
# read_n = 3  # read glimpse grid width/height
# write_n = 1  # write glimpse grid width/height
# write_radius = 3
# read_size = 2 * read_n * read_n if FLAGS.read_attn else 2 * img_size
# write_size = write_n * write_n if FLAGS.write_attn else img_size
# z_size = 100  # QSampler output size
# T = 64  # MNIST generation sequence length
# batch_size = 100  # training minibatch size
# train_iters = 10000
# learning_rate = 1e-3  # learning rate for optimizer
# eps = 1e-8  # epsilon for numerical stability
# draw_with_white = True;  # draw with white ink or black ink

# # ETH
# A, B = 64, 64  # image width,height
# img_size = B * A  # the canvas size
# enc_size = 400  # number of hidden units / output size in LSTM
# dec_size = 400
# read_n = 7  # read glimpse grid width/height
# write_n = 1  # write glimpse grid width/height
# write_radius = 4
# read_size = 2 * read_n * read_n if FLAGS.read_attn else 2 * img_size
# write_size = write_n * write_n if FLAGS.write_attn else img_size
# z_size = 200  # QSampler output size
# T = 150  # MNIST generation sequence length
# batch_size = 100  # training minibatch size
# train_iters = 10000
# learning_rate = 1e-3  # learning rate for optimizer
# eps = 1e-8  # epsilon for numerical stability
# draw_with_white = False;  # draw with white ink or black ink

# # DEBUG
# A, B = 4, 4  # image width,height
# img_size = B * A  # the canvas size
# enc_size = 5  # number of hidden units / output size in LSTM
# dec_size = 5
# read_n = 3  # read glimpse grid width/height
# write_n = 1  # write glimpse grid width/height
# write_radius = 2
# read_size = 2 * read_n * read_n if FLAGS.read_attn else 2 * img_size
# write_size = write_n * write_n if FLAGS.write_attn else img_size
# z_size = 5  # QSampler output size
# T = 5  # MNIST generation sequence length
# batch_size = 100  # training minibatch size
# train_iters = 500
# learning_rate = 1e-3  # learning rate for optimizer
# eps = 1e-8  # epsilon for numerical stability
# draw_with_white = False;  # draw with white ink or black ink

# # BUILD MODEL ## 

DO_SHARE = None  # workaround for variable_scope(reuse=True)

x = tf.placeholder(tf.float32, shape=(config['batch_size'], config['img_size']))  # input (batch_size * img_size)
e = tf.random_normal((config['batch_size'], config['z_size']), mean=0, stddev=1)  # Qsampler noise
lstm_enc = tf.contrib.rnn.LSTMCell(config['enc_size'], state_is_tuple=True)  # encoder Op
lstm_dec = tf.contrib.rnn.LSTMCell(config['dec_size'], state_is_tuple=True)  # decoder Op


def export_config(config, output_file):
  """
  Write the configuration parameters into a human readable file.
  :param config: the configuration dictionary
  :param output_file: the output text file
  """
  if not output_file.endswith('.txt'):
      output_file.append('.txt')
  max_key_length = np.amax([len(k) for k in config.keys()])
  with open(output_file, 'w') as f:
      for k in sorted(config.keys()):
          out_string = '{:<{width}}: {}\n'.format(k, config[k], width=max_key_length)
          f.write(out_string)


def linear(x, output_dim):
  """
  affine transformation Wx+b
  assumes x.shape = (batch_size, num_features)
  """
  w = tf.get_variable("w", [x.get_shape()[1], output_dim])  # , initializer=tf.random_normal_initializer()) 
#   b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
  b = tf.get_variable("b", [output_dim], initializer=tf.random_normal_initializer())
  return tf.matmul(x, w) + b


def filterbank(gx, gy, sigma2, delta, N):
  grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
  mu_x = gx + (grid_i - N / 2 - 0.5) * delta  # eq 19
  mu_y = gy + (grid_i - N / 2 - 0.5) * delta  # eq 20
  a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
  b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
  mu_x = tf.reshape(mu_x, [-1, N, 1])
  mu_y = tf.reshape(mu_y, [-1, N, 1])
  sigma2 = tf.reshape(sigma2, [-1, 1, 1])
  Fx = tf.exp(-tf.square(a - mu_x) / (2 * sigma2))
  Fy = tf.exp(-tf.square(b - mu_y) / (2 * sigma2))  # batch x N x B
  # normalize, sum over A and B dims
  Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keepdims=True), config['eps'])
  Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keepdims=True), config['eps'])
  return Fx, Fy


def read_attn_window(scope, h_dec, x_hat, N):
	with tf.variable_scope(scope, reuse=DO_SHARE):
# 		print(h_dec.shape, x.shape, tf.concat([h_dec, x], 1).shape)
# 		params = linear(tf.concat([h_dec, x_hat], 1), 5)
		params = linear(h_dec, 5)
  # gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
	gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(params, 5, 1)
	gx = (A + 1) / 2 * (gx_ + 1)
	gy = (B + 1) / 2 * (gy_ + 1)
# 	gx = tf.sigmoid(gx_) * A
# 	gy = tf.sigmoid(gy_) * B
	sigma2 = tf.exp(log_sigma2)
	delta = 0 * tf.exp(log_delta)
	if N > 1:
		delta = (max(A, B) - 1) / (N - 1) * tf.exp(log_delta)  # batch x N
	return filterbank(gx, gy, sigma2, delta, N) + (tf.exp(log_gamma), gx, gy, sigma2, delta,)

 
def write_attn_window(scope, h_dec, N):
  with tf.variable_scope(scope, reuse=DO_SHARE):
      params = linear(h_dec, 5)
  # gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
  gx_, gy_, unscaled_sigma2, log_delta, log_gamma = tf.split(params, 5, 1)
  gx = (A + 1) / 2 * (gx_ + 1)
  gy = (B + 1) / 2 * (gy_ + 1)
  sigma2 = tf.sigmoid(unscaled_sigma2) * ((config['write_radius'] / 2) ** 2)
#   sigma2 = tf.exp(unscaled_sigma2)
  delta = 0 * tf.exp(log_delta)
  if N > 1:
		delta = (max(A, B) - 1) / (N - 1) * tf.exp(log_delta)  # batch x N
  return filterbank(gx, gy, sigma2, delta, N) + (tf.exp(log_gamma), gx, gy, sigma2, delta,)


# # READ ## 
def read_no_attn(x, x_hat, h_dec_prev):
  return tf.concat([x, x_hat], 1), tf.concat([0, 0, 0], 1)


def read_attn(x, x_hat, h_dec_prev):
  Fx, Fy, gamma, gx, gy, sigma2, delta = read_attn_window("read", h_dec_prev, x_hat, config['read_n'])

  def filter_img(img, Fx, Fy, gamma, N):
      Fxt = tf.transpose(Fx, perm=[0, 2, 1])
      img = tf.reshape(img, [-1, B, A])
      glimpse = tf.matmul(Fy, tf.matmul(img, Fxt))
      glimpse = tf.reshape(glimpse, [-1, N * N])
      return glimpse * tf.reshape(gamma, [-1, 1])

  x = filter_img(x, Fx, Fy, gamma, config['read_n'])  # batch x (read_n*read_n)
  x_hat = filter_img(x_hat, Fx, Fy, gamma, config['read_n'])
  return tf.concat([x, x_hat], 1), tf.concat([gx, gy, ((config['read_n'] - 1) * delta) + (4.0 * tf.sqrt(sigma2))], 1)  # concat along feature axis


read = read_attn if FLAGS.read_attn else read_no_attn


# # ENCODE ## 
def encode(state, input):
  """
  run LSTM
  state = previous encoder state
  input = cat(read,h_dec_prev)
  returns: (output, new_state)
  """
  with tf.variable_scope("encoder", reuse=DO_SHARE):
      return lstm_enc(input, state)

# # Q-SAMPLER (VARIATIONAL AUTOENCODER) ##


def sampleQ(h_enc):
  """
  Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
  mu is (batch,z_size)
  """
  with tf.variable_scope("mu", reuse=DO_SHARE):
      mu = linear(h_enc, config['z_size'])
  with tf.variable_scope("sigma", reuse=DO_SHARE):
      logsigma = linear(h_enc, config['z_size'])
      sigma = tf.exp(logsigma)
  return (mu + sigma * e, mu, logsigma, sigma)


# # DECODER ## 
def decode(state, input):
  with tf.variable_scope("decoder", reuse=DO_SHARE):
      return lstm_dec(input, state)


# # WRITER ## 
def write_no_attn(h_dec):
  with tf.variable_scope("write", reuse=DO_SHARE):
      return linear(h_dec, config['img_size']), tf.concat([0, 0, 0], 1)


def write_attn(h_dec):
  with tf.variable_scope("writeW", reuse=DO_SHARE):
      w = tf.sigmoid(linear(h_dec, config['write_size']))  # batch x (write_n*write_n)
#   w = tf.ones((batch_size, write_size))
  N = config['write_n']
  w = tf.reshape(w, [config['batch_size'], N, N])
  Fx, Fy, gamma, gx, gy, sigma2, _ = write_attn_window("write", h_dec, config['write_n'])
  Fyt = tf.transpose(Fy, perm=[0, 2, 1])
  wr = tf.matmul(Fyt, tf.matmul(w, Fx))
  wr = tf.reshape(wr, [config['batch_size'], B * A])
  # gamma=tf.tile(gamma,[1,B*A])
  return wr * tf.reshape(1.0 / gamma, [-1, 1]), tf.concat([gx, gy, 4.0 * tf.sqrt(sigma2)], 1)


write = write_attn if FLAGS.write_attn else write_no_attn

# # STATE VARIABLES ## 

cs = [0] * T  # sequence of canvases
scs = [0] * T  # summary canvases
read_bb = [0] * T  # sequence of bounding boxes for reading (center (x,y), (read_n-1)*delta + 4*sigma)
write_bb = [0] * T  # sequence of bounding boxes for writing (center (x,y), 4*sigma)
mus, logsigmas, sigmas = [0] * T, [0] * T, [0] * T  # gaussian params generated by SampleQ. We will need these for computing loss.
# initial states
h_dec_prev = tf.zeros((config['batch_size'], config['dec_size']))
enc_state = lstm_enc.zero_state(config['batch_size'], tf.float32)
dec_state = lstm_dec.zero_state(config['batch_size'], tf.float32)

# # DRAW MODEL ## 

# construct the unrolled computational graph
for t in range(T):
  c_prev = tf.zeros((config['batch_size'], config['img_size'])) if t == 0 else cs[t - 1]
  x_hat = x - tf.tanh(c_prev)  # error image
  r = read(x, x_hat, h_dec_prev)
  h_enc, enc_state = encode(enc_state, tf.concat([r[0], h_dec_prev], 1))
  read_bb[t] = r[1]
  z, mus[t], logsigmas[t], sigmas[t] = sampleQ(h_enc)
  h_dec, dec_state = decode(dec_state, z)
  write_output = write(h_dec)
  cs[t] = c_prev + write_output[0]  # store results
#   scs[t] = tf.image.resize_bicubic(tf.reshape(cs[t][0, :], [B, A]), [100, 100])
  scs[t] = tf.reshape(cs[t][0, :], [B, A])
  write_bb[t] = write_output[1]
  h_dec_prev = h_dec
  DO_SHARE = True  # from now on, share variables
summary_canvas = tf.convert_to_tensor(scs)  # T x B x A
summary_canvas_merged = tf.reshape(summary_canvas, shape=(T * B, A))  # shape=(T * B, A)
# summary_canvas_merged = tf.reshape(summary_canvas, shape=(T * 100, 100))  # shape=(T * B, A)
# tf.summary.image('canvas', tf.reshape(summary_canvas_merged, [1, T * 100, 100, 1]), max_outputs=1)  # [1, T * B, A, 1]
tf.summary.image('canvas', tf.reshape(summary_canvas_merged, [1, T * B, A, 1]), max_outputs=1)  # [1, T * B, A, 1]
# tf.summary.image('reference', tf.reshape(tf.image.resize_images(tf.reshape(x, [B, A]), [100, 100]), [1, 100, 100, 1]), max_outputs=1)  # [1, B, A, 1]
tf.summary.image('reference', tf.reshape(x[0, :], [1, B, A, 1]), max_outputs=1)  # [1, B, A, 1]
# # LOSS FUNCTION ## 


def binary_crossentropy(t, o):
    return -(t * tf.log(o + config['eps']) + (1.0 - t) * tf.log(1.0 - o + config['eps']))


# reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
x_recons = tf.nn.tanh(cs[-1])

# after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
Lx = tf.reduce_sum(binary_crossentropy(x, x_recons), 1)  # reconstruction term
Lx = tf.reduce_mean(Lx)
tf.summary.scalar('Reconstruction Loss', Lx)

kl_terms = [0] * T
for t in range(T):
  mu2 = tf.square(mus[t])
  sigma2 = tf.square(sigmas[t])
  logsigma = logsigmas[t]
  kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - .5  # each kl term is (1xminibatch)
KL = tf.add_n(kl_terms)  # this is 1xminibatch, corresponding to summing kl_terms from 1:T
Lz = tf.reduce_mean(KL)  # average over minibatches
tf.summary.scalar('Latent Loss', Lz)

cost = Lx + Lz
tf.summary.scalar('Total Loss', cost)

# # OPTIMIZER ## 

optimizer = tf.train.AdamOptimizer(config['learning_rate'], beta1=0.5)
grads = optimizer.compute_gradients(cost)
for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
train_op = optimizer.apply_gradients(grads)

# # RUN TRAINING ## 

# # Load Dataset ##


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_gray = tf.image.decode_jpeg(image_string, channels=1)
#   image_gray = image_decoded
#   if image_decoded.get_shape()[-1] == 3:
#   image_gray = tf.image.rgb_to_grayscale(image_decoded)
  image_converted = tf.image.convert_image_dtype(image_gray, tf.float32)
  image_resized = tf.image.resize_images(image_converted, [A, B])
  image_flattened = tf.reshape(image_resized, [-1])
  if not config['draw_with_white']:
    image_flipped = 1.0 - image_flattened
    return image_flipped
  return image_flattened

# MNIST
# data_directory = os.path.join(FLAGS.data_dir, "mnist")
# if not os.path.exists(data_directory):
# 	os.makedirs(data_directory)
# train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data


# CUSTOM
data_directory = os.path.join(FLAGS.data_dir, "train_data")
if not os.path.exists(data_directory):
  print("Train data not found")
  sys.exit()
data_files = tf.gfile.ListDirectory(data_directory)
idx = np.arange(len(data_files))
np.random.shuffle(idx)
data_filenames = [os.path.join(data_directory, data_files[i]) for i in idx]
# data_filenames = data_filenames[:1000]  # DEBUG
dataset = tf.data.Dataset.from_tensor_slices(data_filenames)
dataset = dataset.map(_parse_function)

# dataset = dataset.shuffle(len(data_filenames))
# train_split_size = int(math.floor((0.99 * len(data_filenames))))
# test_split_size = int(math.floor(0.01 * len(data_filenames)))
# train_dataset = dataset.take(train_split_size).repeat().batch(batch_size)
# train_dataset = dataset.shuffle(len(data_filenames)).repeat().batch(batch_size)
train_dataset = dataset.repeat().batch(config['batch_size'])

train_dataset_iterator = train_dataset.make_one_shot_iterator()
next_training_batch = train_dataset_iterator.get_next()

# # SETUP

Lxs = [0] * config['train_iters']
Lzs = [0] * config['train_iters']
sess = tf.InteractiveSession()

timestamp = str(int(time.time()))
config['log_dir'] = os.path.abspath(os.path.join(FLAGS.data_dir, 'logs', 'DRAW' + '_' + timestamp))
os.makedirs(config['log_dir'])
print('Logging data to {}'.format(config['log_dir']))
export_config(config, os.path.join(config['log_dir'], 'config.txt'))
saver = tf.train.Saver()  # saves variables learned during training
merged_summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(config['log_dir'] + '/summary/train', sess.graph)
test_writer = tf.summary.FileWriter(config['log_dir'] + '/summary/test')
tf.global_variables_initializer().run()
# saver.restore(sess, "/tmp/draw/drawmodel.ckpt") # to restore from model, uncomment this line

# # Training
train_fetches = []
test_fetches = []
train_fetches.extend([merged_summaries, Lx, Lz, train_op])
test_fetches.extend([merged_summaries, Lx, Lz])
for i in range(config['train_iters']):
	# xtrain,_=train_data.next_batch(batch_size) # xtrain is (batch_size x img_size)
	xnext = sess.run(next_training_batch)
	feed_dict = {x:xnext}
	if i % 100 == 0:
		xlog = xnext
		summary, Lxs[i], Lzs[i] = sess.run(test_fetches, feed_dict)
		print("iter=%d : Lx: %f Lz: %f" % (i, Lxs[i], Lzs[i]))
		test_writer.add_summary(summary, i)
	else:
		summary, Lxs[i], Lzs[i], _ = sess.run(train_fetches, feed_dict)
		if i % 100 == 1:
			train_writer.add_summary(summary, i)

# # TRAINING FINISHED ## 

# Testing
# test_dataset = dataset.skip(train_split_size).batch(batch_size)
# test_dataset_iterator = test_dataset.make_one_shot_iterator()
# next_testing_batch = test_dataset_iterator.get_next()
# num_test_batches = int(math.floor(test_split_size / batch_size))
# fetches = []
# fetches.extend([Lx, Lz])
# Lx_test = 0.0
# Lz_test = 0.0
# for i in range(num_test_batches):
# 	xtest = sess.run(next_testing_batch)
# 	feed_dict = {x:xtest}
# 	results = sess.run(fetches, feed_dict)
# 	Lx_test_i, Lz_test_i = results
# 	Lx_test += Lx_test_i + Lx_test_i
# 	Lz_test += Lz_test_i + Lz_test_i
# Lx_test /= num_test_batches
# Lz_test /= num_test_batches
# print("Test dataset results : Lx: %f Lz: %f" % (Lx_test, Lz_test))

# # Logging + Visualization
# log_dataset = test_dataset.take(batch_size)
# batched_log_dataset = log_dataset.batch(batch_size)
# log_dataset_iterator = batched_log_dataset.make_one_shot_iterator()
# log_batch = log_dataset_iterator.get_next()
# xlog = sess.run(log_batch)

feed_dict = {x:xlog}

canvases, read_bounding_boxes, write_bounding_boxes = sess.run([cs, read_bb, write_bb], feed_dict)  # generate some examples
canvases = np.array(canvases)  # T x batch x img_size
read_bounding_boxes = np.array(read_bounding_boxes)  # T x batch x 3
write_bounding_boxes = np.array(write_bounding_boxes)  # T x batch x 3

out_file = os.path.join(config['log_dir'], "draw_data.npy")
np.save(out_file, [xlog, canvases, read_bounding_boxes, write_bounding_boxes, Lxs, Lzs, config['draw_with_white']])
print("Outputs saved in file: %s" % out_file)

ckpt_file = os.path.join(config['log_dir'], "drawmodel.ckpt")
print("Model saved in file: %s" % saver.save(sess, ckpt_file))

sess.close()
