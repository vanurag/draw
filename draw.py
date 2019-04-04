#!/usr/bin/env python

""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow

Example Usage: 
	python draw.py --data_dir=/tmp/draw --log_dir=/tmp/draw/logs

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
tf.flags.DEFINE_string("log_dir", "", "")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn", True, "enable attention for writer")
FLAGS = tf.flags.FLAGS
A = config['A']
B = config['B']
T = config['T']

# # BUILD MODEL ## 

DO_SHARE = None  # workaround for variable_scope(reuse=True)

x = tf.placeholder(tf.float32, shape=(config['batch_size'], config['img_size']))  # input (batch_size * img_size)
e = tf.random_normal((config['batch_size'], config['z_size']), mean=0, stddev=1)  # Qsampler noise


def get_lstm_cell(rnn_mode, hidden_size):
  if rnn_mode == "BASIC":
    return tf.contrib.rnn.BasicLSTMCell(
        hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=DO_SHARE)
  if rnn_mode == "BLOCK":
    return tf.contrib.rnn.LSTMBlockCell(
        hidden_size, forget_bias=0.0, reuse=DO_SHARE)
  if rnn_mode == "GRU":
    return tf.contrib.rnn.GRUBlockCellV2(
        hidden_size, reuse=DO_SHARE)
  raise ValueError("rnn_mode %s not supported" % rnn_mode)

# def make_rnn_cell(l):
#   cell = get_lstm_cell()
#   if self.is_training and self.keep_prob < 1 and l not in self.config['no_dropout_ids']:
#     cell = tf.contrib.rnn.DropoutWrapper(
#         cell, output_keep_prob=self.keep_prob)
#           if self.use_residual:
#             cell = tf.contrib.rnn.ResidualWrapper(cell)
#   return cell


lstm_enc = tf.contrib.rnn.MultiRNNCell(
  [get_lstm_cell(config['enc_rnn_mode'], config['enc_size']) for l in range(config['n_enc_layers'])], state_is_tuple=True)  # encoder Op
lstm_dec = tf.contrib.rnn.MultiRNNCell(
  [get_lstm_cell(config['dec_rnn_mode'], config['dec_size']) for l in range(config['n_dec_layers'])], state_is_tuple=True)  # encoder Op
          
# lstm_enc = tf.contrib.rnn.LSTMCell(config['enc_size'], state_is_tuple=True)  # encoder Op
# lstm_dec = tf.contrib.rnn.LSTMCell(config['dec_size'], state_is_tuple=True)  # decoder Op


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
  if config['draw_with_white']:
    scs[t] = tf.transpose(tf.reshape(cs[t][:config['n_summary_per_batch'], :], [config['n_summary_per_batch'], B, A]), perm=[1, 0, 2])  # B x batch_size x A
  else:
    scs[t] = tf.transpose(tf.reshape(1 - cs[t][:config['n_summary_per_batch'], :], [config['n_summary_per_batch'], B, A]), perm=[1, 0, 2])  # B x batch_size x A
  write_bb[t] = write_output[1]
  h_dec_prev = h_dec
  DO_SHARE = True  # from now on, share variables
summary_canvas = tf.convert_to_tensor(scs)  # T x B x batch_size x A
summary_canvas_merged = tf.reshape(summary_canvas, shape=(T * B, config['n_summary_per_batch'] * A))  # shape=(T * B, batch_size * A)
tf.summary.image('canvas', tf.reshape(summary_canvas_merged, [1, T * B, config['n_summary_per_batch'] * A, 1]), max_outputs=1)  # [1, T * B, batch_size * A, 1]
if config['draw_with_white']:
	sx = tf.transpose(tf.reshape(x[:config['n_summary_per_batch'], :], [config['n_summary_per_batch'], B, A]), perm=[1, 0, 2])
else:
	sx = tf.transpose(tf.reshape(1 - x[:config['n_summary_per_batch'], :], [config['n_summary_per_batch'], B, A]), perm=[1, 0, 2])
tf.summary.image('reference', tf.reshape(sx, [1, B, config['n_summary_per_batch'] * A, 1]), max_outputs=1)  # [1, B, A, 1]

# # LOSS FUNCTION ## 


def binary_crossentropy(t, o):
    return -(t * tf.log(o + config['eps']) + (1.0 - t) * tf.log(1.0 - o + config['eps']))


anchor_point = int(T / 2)
# reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
x_recons = tf.nn.tanh(cs[-1])
x_recons_anchor = tf.nn.tanh(cs[anchor_point])

# after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
Lx_end = tf.reduce_sum(binary_crossentropy(x, x_recons), 1)  # reconstruction term
Lx_end = tf.reduce_mean(Lx_end)
# Lx_anchor = tf.reduce_sum(binary_crossentropy(tf.multiply(x, x_recons_anchor), x_recons_anchor), 1)  # reconstruction term
Lx_anchor = tf.reduce_sum(binary_crossentropy(x, x_recons_anchor), 1)  # reconstruction term
Lx_anchor = tf.reduce_mean(Lx_anchor)
Lx = Lx_end  # + 0.2 * Lx_anchor
tf.summary.scalar('Reconstruction Loss at anchor', Lx_anchor)
tf.summary.scalar('Reconstruction Loss at end', Lx_end)
tf.summary.scalar('Total Reconstruction Loss', Lx)

kl_terms = [0] * T
for t in range(T):
  mu2 = tf.square(mus[t])
  sigma2 = tf.square(sigmas[t])
  logsigma = logsigmas[t]
  kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - .5  # each kl term is (1xminibatch)
KL = tf.add_n(kl_terms)  # this is 1xminibatch, corresponding to summing kl_terms from 1:T
Lz = tf.reduce_mean(KL)  # average over minibatches
tf.summary.scalar('Latent Loss', Lz)

cost = Lx + 0.1 * Lz
tf.summary.scalar('Total Loss', cost)

# # OPTIMIZER ## 

global_step = tf.Variable(1, name='global_step', trainable=False)

# configure learning rate
if config['learning_rate_type'] == 'exponential':
  lr = tf.train.exponential_decay(config['learning_rate'],
                                  global_step=global_step,
                                  decay_steps=config['learning_rate_decay_steps'],
                                  decay_rate=config['learning_rate_decay_rate'],
                                  staircase=False)
  lr_decay_op = tf.identity(lr)
elif config['learning_rate_type'] == 'linear':
  lr = tf.Variable(config['learning_rate'], trainable=False)
  lr_decay_op = lr.assign(tf.multiply(lr, config['learning_rate_decay_rate']))
elif config['learning_rate_type'] == 'fixed':
  lr = config['learning_rate']
  lr_decay_op = tf.identity(lr)
else:
  raise ValueError('learning rate type "{}" unknown.'.format(config['learning_rate_type']))
tf.summary.scalar('learning_rate', lr)

optimizer = tf.train.AdamOptimizer(config['learning_rate'], beta1=0.5)
grads = optimizer.compute_gradients(cost)
for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
train_op = optimizer.apply_gradients(grads, global_step=global_step)

# # RUN TRAINING ## 

# # Load Dataset ##


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_gray = tf.image.decode_jpeg(image_string, channels=1)
  image_converted = tf.image.convert_image_dtype(image_gray, tf.float32)
  image_resized = tf.image.resize_images(image_converted, [A, B])
  image_flattened = tf.reshape(image_resized, [-1])
  if not config['draw_with_white']:
    image_flipped = 1.0 - image_flattened
    return image_flipped
  return image_flattened


# CUSTOM
data_directory = FLAGS.data_dir
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
train_dataset = dataset.repeat().batch(config['batch_size'])
train_dataset_iterator = train_dataset.make_one_shot_iterator()
next_training_batch = train_dataset_iterator.get_next()

# # SETUP

Lxs = [0] * config['train_iters']
Lzs = [0] * config['train_iters']
costs = [0] * config['train_iters']
sess = tf.InteractiveSession()

timestamp = str(int(time.time()))
config['log_dir'] = os.path.abspath(os.path.join(FLAGS.log_dir, 'DRAW' + '_' + timestamp))
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
lowest_test_loss = 1.0e6
last_saved_epoch = 0  # epoch corresponding to last saved chkpnt
train_fetches = []
test_fetches = []
train_fetches.extend([merged_summaries, Lx, Lz, cost, train_op])
test_fetches.extend([merged_summaries, Lx, Lz, cost])
for i in range(config['train_iters']):
	step = tf.train.global_step(sess, global_step)
	if config['learning_rate_type'] == 'linear' and i % config['learning_rate_decay_steps'] == 0:
		sess.run(lr_decay_op)
	xnext = sess.run(next_training_batch)
	feed_dict = {x:xnext}
	if i % 100 == 0:
		xlog = xnext
		summary, Lxs[i], Lzs[i], costs[i] = sess.run(test_fetches, feed_dict)
		print("iter=%d : Lx: %f Lz: %f cost: %f" % (i, Lxs[i], Lzs[i], costs[i]))
		test_writer.add_summary(summary, global_step=step)
		# save this checkpoint if necessary
		if (i - last_saved_epoch + 1) >= config['save_checkpoints_every_epoch'] and costs[i] < lowest_test_loss:
			last_saved_epoch = i
			lowest_test_loss = costs[i]
			saver.save(sess, os.path.join(config['log_dir'], 'drawmodel'), i)
	else:
		summary, Lxs[i], Lzs[i], _, _ = sess.run(train_fetches, feed_dict)
		if i % 100 == 1:
			train_writer.add_summary(summary, global_step=step)

# # TRAINING FINISHED ## 

# # Logging + Visualization
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
