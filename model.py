import tensorflow as tf
import math
import sys
from ops import *
from matplotlib.pyplot import axis

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d


class DrawModel(object):
  """
  Creates training and validation computational graphs.
  Note that tf.variable_scope enables parameter sharing so that both graphs are identical.
  """
  
  def __init__(self, config, placeholders, mode, discriminator=None, annealing_schedules=None):
    """
    Basic setup.
    :param config: configuration dictionary
    :param placeholders: dictionary of input placeholders
    :param mode: training, validation or inference
    """
    assert mode in ['training', 'validation', 'inference']
    self.config = config
    self.input_ = placeholders['input_pl']
    self.start_canvas_ = placeholders['canvas_pl']
    self.mode = mode
    self.DO_SHARE = True if self.mode == 'validation' else None
    self.A = config['A']  # image width
    self.B = config['B']  # image height
    self.img_size = config['img_size']  # the canvas size
    self.draw_with_white = config['draw_with_white']  # draw with white ink or black ink
    self.draw_all_time = config['draw_all_time']  # should draw for all T time steps or end if the result looks close to target
    
    self.enc_rnn_mode = config['enc_rnn_mode']  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
    self.enc_size = config['enc_size']  # number of hidden units / output size in LSTM layer
    self.n_enc_layers = config['n_enc_layers']  # number of layers in encoder LSTM
    self.dec_rnn_mode = config['dec_rnn_mode']  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
    self.dec_size = config['dec_size'] 
    self.n_dec_layers = config['n_dec_layers']  # number of layers in decoder LSTM
    self.z_size = config['z_size']  # QSampler output size
    # generation sequence length  
    if self.mode == 'inference':
      self.T = config['T']  # fixed for inference
    else:
      self.T = tf.Variable(config['T'], trainable=False)  # variable during training and validation
    
    self.read_n = config['read_n']  # read glimpse grid width/height
    self.write_n = config['write_n']  # write glimpse grid width/height
    self.write_radius = config['write_radius'] 
    self.use_read_attn = config['read_attn'] 
    self.read_size = config['read_size'] 
    self.use_write_attn = config['write_attn']
    self.write_size = config['write_size']
    self.write_decision_temperature = 1.0 
    self.write_decision_prior_log_odds = -0.01
    # Higher threshold -> longer draw times
    if self.mode == 'inference':
      self.stop_writing_threshold = 0.99
    else:
      self.stop_writing_threshold = tf.Variable(0.99, trainable=False)
    
    self.batch_size = config['batch_size']  # training minibatch size
    self.n_summary_per_batch = config['n_summary_per_batch'] 

    self.eps = config['eps']  # epsilon for numerical stability

    self.summary_collection = 'training_summaries' if mode == 'training' else 'validation_summaries'
    
    self.e = tf.random_normal((config['batch_size'], config['z_size']), mean=0, stddev=1)  # Qsampler noise
    
#     if discriminator is None and self.mode is not 'inference':
#       print('Please provide a Discriminator object!')
#       sys.exit()
    self.discriminator = discriminator
    self.df_mode = config['disc_mode']
    
    # Create a variable that stores how many training iterations we performed.
    # This is useful for saving/storing the network
    if self.mode is not 'inference':
      self.global_step = tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False)
      
      # Learning rate during training
      self.learning_rate = tf.Variable(1e-4, trainable=False)
    
    if annealing_schedules is not None:
      for param, schedule in annealing_schedules.items():
        # replacing some of the parameters by annealed
        # versions, if schedule is provided for those
        setattr(self, param, self._create_annealed_tensor(
            param, schedule, self.global_step
        ))
        tf.summary.scalar(param, getattr(self, param), collections=[self.summary_collection], family='annealing')
    
    # encoder Op
    if self.n_enc_layers > 1:
      self.lstm_enc = tf.contrib.rnn.MultiRNNCell(
        [self.get_lstm_cell(self.enc_rnn_mode, self.enc_size) for l in range(self.n_enc_layers)], state_is_tuple=True)
    else:
      self.lstm_enc = self.get_lstm_cell(self.enc_rnn_mode, self.enc_size)
    # Decoder Op
    if self.n_dec_layers > 1:
      self.lstm_dec = tf.contrib.rnn.MultiRNNCell(
        [self.get_lstm_cell(self.dec_rnn_mode, self.dec_size) for l in range(self.n_dec_layers)], state_is_tuple=True)
    else:
      self.lstm_dec = self.get_lstm_cell(self.dec_rnn_mode, self.dec_size)
    
    # # STATE VARIABLES ## 
#     self.cs = [0] * self.T  # sequence of canvases
    self.cs = tf.TensorArray(dtype=tf.float32, size=self.T, dynamic_size=False,
                             element_shape=tf.TensorShape([self.batch_size, self.img_size]),
                             clear_after_read=False)
    # summary of canvases B x batch_size x A
    self.scs = tf.TensorArray(dtype=tf.float32, size=self.T, dynamic_size=False,
                              element_shape=tf.TensorShape([self.B, self.n_summary_per_batch, self.A]))
#     self.read_bb = [0] * self.T  # sequence of bounding boxes for reading (center (x,y), (read_n-1)*delta + 4*sigma)
    self.read_bb = tf.TensorArray(dtype=tf.float32, size=self.T, dynamic_size=False,
                                  element_shape=tf.TensorShape([self.batch_size, 3]))
#     self.write_bb = [0] * self.T  # sequence of bounding boxes for writing (center (x,y), 4*sigma, write_intensity)
    self.write_bb = tf.TensorArray(dtype=tf.float32, size=self.T, dynamic_size=False,
                                   element_shape=tf.TensorShape([self.batch_size, 4]),
                                   clear_after_read=False)
#     self.mus, self.logsigmas, self.sigmas = [0] * self.T, [0] * self.T, [0] * self.T  # gaussian params generated by SampleQ. We will need these for computing loss.
    self.mus = tf.TensorArray(dtype=tf.float32, size=self.T, dynamic_size=False,
                              element_shape=tf.TensorShape([self.batch_size, self.z_size]))
    self.logsigmas = tf.TensorArray(dtype=tf.float32, size=self.T, dynamic_size=False,
                                    element_shape=tf.TensorShape([self.batch_size, self.z_size]))
    self.sigmas = tf.TensorArray(dtype=tf.float32, size=self.T, dynamic_size=False,
                                 element_shape=tf.TensorShape([self.batch_size, self.z_size]))
    # Probability of whether to write or not
    self.should_write_log_odds = tf.TensorArray(dtype=tf.float32, size=self.T, dynamic_size=False,
                                                element_shape=tf.TensorShape([self.batch_size, 1]))
    self.should_write_pre_sigmoid = tf.TensorArray(dtype=tf.float32, size=self.T, dynamic_size=False,
                                                   element_shape=tf.TensorShape([self.batch_size, 1]))
    self.should_write_decision = tf.TensorArray(dtype=tf.float32, size=self.T, dynamic_size=False,
                                                element_shape=tf.TensorShape([self.batch_size, 1]))
    self.draw_intensity = tf.TensorArray(dtype=tf.float32, size=self.T, dynamic_size=False,
                                         element_shape=tf.TensorShape([self.batch_size, 1]))
    
    self.stop_times = tf.zeros(self.batch_size, dtype=tf.int32)
      
    self.read = self.read_attn if self.use_read_attn else self.read_no_attn
    self.write = self.write_attn if self.use_write_attn else self.write_no_attn
    
  @staticmethod
  def _create_annealed_tensor(param, schedule, global_step, eps=10e-10):
    if schedule["decay_type"] == "linear":
      if "staircase" not in schedule or not schedule["staircase"]:
        value = schedule["init"] + (schedule["factor"] * (tf.cast(global_step, tf.float32) / schedule["iters"]))
      else:
        value = schedule["init"] + (schedule["factor"] * tf.floor((tf.cast(global_step, tf.float32) / schedule["iters"])))
    elif schedule["decay_type"] == "exponential":
      value = tf.train.exponential_decay(
        learning_rate=schedule["init"], global_step=global_step,
        decay_steps=schedule["iters"], decay_rate=schedule["factor"],
        staircase=False if "staircase" not in schedule else schedule["staircase"],
        name=param
      )
    elif schedule["decay_type"] == "fixed":
      value = schedule["init"]
    else:
      raise ValueError('Unknow decay type "{}" provided.'.format(schedule["decay_type"]))
    
    if "min" in schedule:
      value = tf.maximum(
          value, schedule["min"],
          name=param + "_max"
      )

    if "max" in schedule:
      value = tf.minimum(
          value, schedule["max"],
          name=param + "_min"
      )

    if "log" in schedule and schedule["log"]:
      value = tf.log(
          value + eps,
          name=param + "_log"
      )

    return value
    
  def build_graph(self):
    self.build_model()
    self.build_loss()
    self.build_optim()
    self.count_parameters()
      
  def get_lstm_cell(self, rnn_mode, hidden_size):
    if rnn_mode == "BASIC":
      return tf.contrib.rnn.LSTMCell(
          hidden_size, state_is_tuple=True)
    if rnn_mode == "BLOCK":
      return tf.contrib.rnn.LSTMBlockCell(
          hidden_size, reuse=self.DO_SHARE)
    if rnn_mode == "GRU":
      return tf.contrib.rnn.GRUBlockCellV2(
          hidden_size, reuse=self.DO_SHARE)
    raise ValueError("rnn_mode %s not supported" % rnn_mode)
  
  def binary_crossentropy(self, t, o):
    return -(t * tf.log(o + self.eps) + (1.0 - t) * tf.log(1.0 - o + self.eps))
  
  def l2_loss(self, t, o):
    return (t - o) ** 2
  
  def filterbank(self, gx, gy, sigma2, delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta  # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta  # eq 20
    a = tf.reshape(tf.cast(tf.range(self.A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(self.B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square(a - mu_x) / (2 * sigma2))
    Fy = tf.exp(-tf.square(b - mu_y) / (2 * sigma2))  # batch x N x B
    # normalize, sum over A and B dims
    Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keepdims=True), self.eps)
    Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keepdims=True), self.eps)
    return Fx, Fy
  
  def read_attn_window(self, scope, h_dec, x_hat, N):
    with tf.variable_scope(scope, reuse=self.DO_SHARE):
  #     print(h_dec.shape, x.shape, tf.concat([h_dec, x], 1).shape)
  #     params = linear(tf.concat([h_dec, x_hat], 1), 5)
      params = linear2(h_dec, 5, hidden_layer_size=self.config['n_hidden_units'])
    # gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
    gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(params, 5, 1)
    gx = (self.A + 1) / 2 * (gx_ + 1)
    gy = (self.B + 1) / 2 * (gy_ + 1)
  #   gx = tf.sigmoid(gx_) * A
  #   gy = tf.sigmoid(gy_) * B
    sigma2 = tf.exp(log_sigma2)
    delta = 0 * tf.exp(log_delta)
    if N > 1:
      delta = (max(self.A, self.B) - 1) / (N - 1) * tf.exp(log_delta)  # batch x N
    return self.filterbank(gx, gy, sigma2, delta, N) + (tf.exp(log_gamma), gx, gy, sigma2, delta,)
   
  def write_attn_window(self, scope, h_dec, N):
    with tf.variable_scope(scope, reuse=self.DO_SHARE):
        params = linear2(h_dec, 5, hidden_layer_size=self.config['n_hidden_units'])
    # gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
    gx_, gy_, unscaled_sigma2, log_delta, log_gamma = tf.split(params, 5, 1)
    gx = (self.A + 1) / 2 * (gx_ + 1)
    gy = (self.B + 1) / 2 * (gy_ + 1)
    sigma2 = tf.sigmoid(unscaled_sigma2) * ((self.write_radius / 2) ** 2)
  #   sigma2 = tf.exp(unscaled_sigma2)
    delta = 0 * tf.exp(log_delta)
    if N > 1:
      delta = (max(self.A, self.B) - 1) / (N - 1) * tf.exp(log_delta)  # batch x N
    return self.filterbank(gx, gy, sigma2, delta, N) + (tf.exp(log_gamma), gx, gy, sigma2, delta,)
  
  # # READ ## 
  def read_no_attn(self, x, x_hat, h_dec_prev):
    return tf.concat([x, x_hat], 1), tf.concat([0, 0, 0], 1)
  
  def read_attn(self, x, x_hat, h_dec_prev):
    Fx, Fy, gamma, gx, gy, sigma2, delta = self.read_attn_window("read", h_dec_prev, x_hat, self.read_n)
  
    def filter_img(img, Fx, Fy, gamma, N):
        Fxt = tf.transpose(Fx, perm=[0, 2, 1])
        img = tf.reshape(img, [-1, self.B, self.A])
        glimpse = tf.matmul(Fy, tf.matmul(img, Fxt))
        glimpse = tf.reshape(glimpse, [-1, N * N])
        return glimpse * tf.reshape(gamma, [-1, 1])
  
    x = filter_img(x, Fx, Fy, gamma, self.read_n)  # batch x (read_n*read_n)
    x_hat = filter_img(x_hat, Fx, Fy, gamma, self.read_n)
    return tf.concat([x, x_hat], 1), tf.concat([gx, gy, ((self.read_n - 1) * delta) + (4.0 * tf.sqrt(sigma2))], 1)  # concat along feature axis
  
  # # ENCODE ## 
  def encode(self, state, input):
    """
    run LSTM
    state = previous encoder state
    input = cat(read,h_dec_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("encoder", reuse=self.DO_SHARE):
        return self.lstm_enc(input, state)
  
  # # Q-SAMPLER (VARIATIONAL AUTOENCODER) ##
  
  def sampleQ(self, h_enc):
    """
    Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
    mu is (batch,z_size)
    """
    with tf.variable_scope("mu", reuse=self.DO_SHARE):
        mu = linear2(h_enc, self.z_size, hidden_layer_size=self.config['n_hidden_units'])
    with tf.variable_scope("sigma", reuse=self.DO_SHARE):
        logsigma = linear2(h_enc, self.z_size, hidden_layer_size=self.config['n_hidden_units'])
        sigma = tf.exp(logsigma)
    return (mu + sigma * self.e, mu, logsigma, sigma)
  
  # # DECODER ## 
  def decode(self, state, input):
    with tf.variable_scope("decoder", reuse=self.DO_SHARE):
        return self.lstm_dec(input, state)
  
  # # WRITER ## 
  def write_no_attn(self, h_dec):
    with tf.variable_scope("write", reuse=self.DO_SHARE):
        return linear2(h_dec, self.img_size, hidden_layer_size=self.config['n_hidden_units']), tf.concat([0, 0, 0], 1)
  
  def write_attn(self, h_dec):
    with tf.variable_scope("writeW", reuse=self.DO_SHARE):
        w1 = tf.exp(linear2(h_dec, self.write_size, hidden_layer_size=self.config['n_hidden_units']))  # batch x (write_n*write_n)
  #   w = tf.ones((batch_size, write_size))
    N = self.write_n
    w = tf.reshape(w1, [self.batch_size, N, N])
    Fx, Fy, gamma, gx, gy, sigma2, _ = self.write_attn_window("write", h_dec, self.write_n)
    Fyt = tf.transpose(Fy, perm=[0, 2, 1])
    wr = tf.matmul(Fyt, tf.matmul(w, Fx))
    wr = tf.reshape(wr, [self.batch_size, self.B * self.A])
    # gamma=tf.tile(gamma,[1,B*A])
    return wr, tf.concat([gx, gy, 4.0 * tf.sqrt(sigma2), w1], 1)
  
  def write_decision(self, h_dec):

    def _concrete_binary_pre_sigmoid_sample(log_odds, temperature, eps=10e-10):
      count = tf.shape(log_odds)[0]
      u = tf.random_uniform([count, 1], minval=0, maxval=1)
      noise = tf.log(u + eps) - tf.log(1.0 - u + eps)
      y = (log_odds + noise) / temperature
      return y

    with tf.variable_scope("write_decision", reuse=self.DO_SHARE):
#       sw_log_odds = linear2(h_dec, 1, hidden_layer_size=0, bias_initializer=tf.constant_initializer(10, tf.float32))
      sw_log_odds = linear2(h_dec, 1, hidden_layer_size=self.config['n_hidden_units'])
      
    sw_pre_sigmoid = _concrete_binary_pre_sigmoid_sample(sw_log_odds, self.write_decision_temperature)
    sw = tf.sigmoid(sw_pre_sigmoid)
    if not self.mode == 'training':
      sw = tf.round(sw)
      
    return sw, sw_log_odds, sw_pre_sigmoid
  
  def draw_loop_body(self, t, stop_sum, should_write_decision, stop_times, cs, scs, read_bb, write_bb, mus, logsigmas, sigmas,
                     should_write_log_odds, should_write_pre_sigmoid, h_dec_prev, enc_state, dec_state):
    if self.draw_all_time:
      stop_times += tf.ones(self.batch_size, dtype=tf.int32)
    else:
      stop_times += tf.where(tf.less(stop_sum, self.stop_writing_threshold),
                             tf.ones(self.batch_size, dtype=tf.int32), tf.zeros(self.batch_size, dtype=tf.int32))
    c_prev = tf.cond(tf.equal(t, 0), lambda: self.start_canvas_, lambda: cs.read(t - 1))
    x_hat = self.input_ - tf.tanh(c_prev)  # error image
    r = self.read(self.input_, x_hat, h_dec_prev)
    h_enc, enc_state = self.encode(enc_state, tf.concat([r[0], h_dec_prev], 1))
    read_bb = read_bb.write(t, r[1])
    z, mu, logsigma, sigma = self.sampleQ(h_enc)
    mus = mus.write(t, mu)
    logsigmas = logsigmas.write(t, logsigma)
    sigmas = sigmas.write(t, sigma)
    h_dec, dec_state = self.decode(dec_state, z)
    write_output = self.write(h_dec)
    sw, sw_log_odd, sw_pre_sigmoid = self.write_decision(h_dec)
    if self.draw_all_time:
      cs = cs.write(t, c_prev + write_output[0])
    else:
#       cs = cs.write(
#         t, tf.where(tf.less(stop_sum, self.stop_writing_threshold),
#                     c_prev + tf.tile(sw, [1, self.img_size]) * write_output[0], tf.zeros_like(c_prev)))
      cs = cs.write(
        t, tf.where(tf.less(stop_sum, self.stop_writing_threshold),
                    c_prev + write_output[0], tf.zeros_like(c_prev)))
    should_write_log_odds = should_write_log_odds.write(t, sw_log_odd)
    should_write_pre_sigmoid = should_write_pre_sigmoid.write(t, sw_pre_sigmoid)
    stop_sum += 1.0 - tf.reshape(sw, [self.batch_size])
    should_write_decision = should_write_decision.write(t, sw)
#     if not self.mode == 'training':
#       should_write_decision = should_write_decision.write(t, tf.round(tf.nn.sigmoid(sw_log_odd)))
#     else:
#       should_write_decision = should_write_decision.write(t, tf.nn.sigmoid(sw_log_odd))
    if self.draw_with_white:
      scs = scs.write(t, tf.transpose(\
        tf.reshape(tf.nn.tanh(cs.read(t)[:self.n_summary_per_batch, :]), \
                   [self.n_summary_per_batch, self.B, self.A]), perm=[1, 0, 2]))  # B x batch_size x A
    else:
      scs = scs.write(t, tf.transpose(\
        tf.reshape(1 - tf.nn.tanh(cs.read(t)[:self.n_summary_per_batch, :]), \
                   [self.n_summary_per_batch, self.B, self.A]), perm=[1, 0, 2]))  # B x batch_size x A
    write_bb = write_bb.write(t, write_output[1])
    h_dec_prev = h_dec
    self.DO_SHARE = True  # from now on, share variables
    return [tf.add(t, 1), stop_sum, should_write_decision, stop_times, cs, scs, read_bb, write_bb, mus, logsigmas, sigmas,
            should_write_log_odds, should_write_pre_sigmoid, h_dec_prev, enc_state, dec_state]
  
  def draw_loop_cond(self, t, stop_sum, *_):
#     return tf.logical_and(
#       tf.less(t, self.T),
#       tf.reduce_any(tf.less(stop_sum, self.stop_writing_threshold)))
    return tf.less(t, self.T)
  
  def build_model(self):
    """
    Builds the actual model.
    """
    with tf.variable_scope('draw_model'):  # , reuse=self.DO_SHARE):
      # initial states
      h_dec_prev = tf.zeros((self.batch_size, self.dec_size))
      enc_state = self.lstm_enc.zero_state(self.batch_size, tf.float32)
      dec_state = self.lstm_dec.zero_state(self.batch_size, tf.float32)
      
      tf.summary.scalar('max_draw_time', self.T, collections=[self.summary_collection], family='variables')
      
      # Generator
      t = tf.constant(0);
      stop_sum = tf.zeros(self.batch_size)
      t, stop_sum, self.should_write_decision, self.stop_times, self.cs, self.scs, self.read_bb, self.write_bb, self.mus, self.logsigmas, self.sigmas, \
      self.should_write_log_odds, self.should_write_pre_sigmoid, h_dec_prev, enc_state, dec_state = \
        tf.while_loop(self.draw_loop_cond, self.draw_loop_body,
                      [t, stop_sum, self.should_write_decision, self.stop_times, self.cs, self.scs, self.read_bb, self.write_bb, self.mus, self.logsigmas, self.sigmas, \
                       self.should_write_log_odds, self.should_write_pre_sigmoid, h_dec_prev, enc_state, dec_state],
                      parallel_iterations=1)
        
      summary_decisions = tf.transpose(self.should_write_decision.stack(), perm=[1, 0, 2])[:self.n_summary_per_batch, :, :]
      tf.summary.image('Write_Decision', \
                       tf.reshape(summary_decisions, shape=(1, self.n_summary_per_batch, self.config['T'], 1)), max_outputs=1, \
                       collections=[self.summary_collection])
        
      summary_canvas = self.scs.stack()  # T x B x batch_size x A
      summary_canvas_merged = tf.reshape(summary_canvas, shape=(self.T * self.B, self.n_summary_per_batch * self.A))  # shape=(T * B, batch_size * A)
      tf.summary.image('canvas', \
                       tf.reshape(summary_canvas_merged, \
                                  [1, self.T * self.B, self.n_summary_per_batch * self.A, 1]), max_outputs=1, \
                       collections=[self.summary_collection])  # [1, T * B, batch_size * A, 1]
      if self.draw_with_white:
        sx = tf.transpose(\
          tf.reshape(self.input_[:self.n_summary_per_batch, :], \
                     [self.n_summary_per_batch, self.B, self.A]), perm=[1, 0, 2])
      else:
        sx = tf.transpose(\
          tf.reshape(1 - self.input_[:self.n_summary_per_batch, :], \
                     [self.n_summary_per_batch, self.B, self.A]), perm=[1, 0, 2])
      tf.summary.image('reference', \
                       tf.reshape(sx, [1, self.B, self.n_summary_per_batch * self.A, 1]), max_outputs=1, \
                       collections=[self.summary_collection])  # [1, B, A, 1]
  
  def build_loss(self):
    """
    Builds the loss function.
    """
    # only need loss if we are not in inference mode
    if self.mode is not 'inference':
      with tf.name_scope('loss'):

        # Reconstruction result
        def _reconstruction_loop_body(t, stop_times, cs, x_recons):
          x_recons = tf.where(tf.less(t, stop_times), tf.nn.tanh(cs.read(t)), x_recons)
          return [tf.add(t, 1), stop_times, cs, x_recons]
          
        self.x_recons = tf.nn.tanh(self.cs.read(0))
        t = tf.constant(0)
        t, self.stop_times, self.cs, self.x_recons = \
          tf.while_loop(lambda t, *_: tf.less(t, self.T), _reconstruction_loop_body,
                        [t, self.stop_times, self.cs, self.x_recons], parallel_iterations=1)
          
        if self.draw_with_white:
          sx_recons = tf.transpose(
            tf.reshape(self.x_recons[:self.n_summary_per_batch, :],
                       [self.n_summary_per_batch, self.B, self.A]), perm=[1, 0, 2])
        else:
          sx_recons = tf.transpose(
            tf.reshape(1 - self.x_recons[:self.n_summary_per_batch, :],
                       [self.n_summary_per_batch, self.B, self.A]), perm=[1, 0, 2])
        tf.summary.image('result',
                         tf.reshape(sx_recons, [1, self.B, self.n_summary_per_batch * self.A, 1]), max_outputs=1,
                         collections=[self.summary_collection])
        
        #############################################################
        # Reconstruction loss - Cross entropy
        #############################################################
        Lx_end_bce = tf.reduce_sum(self.binary_crossentropy(self.input_, self.x_recons), 1)
        Lx_end_bce = tf.reduce_mean(Lx_end_bce)
        tf.summary.scalar('Reconstruction Loss (BCE)', Lx_end_bce, collections=[self.summary_collection], family='loss')
        
        #############################################################
        # Reconstruction loss - L2
        #############################################################
        Lx_end_l2 = tf.reduce_sum(self.l2_loss(self.input_, self.x_recons), 1)
        Lx_end_l2 = 100.0 * tf.reduce_mean(Lx_end_l2)
        tf.summary.scalar('Reconstruction Loss (L2)', Lx_end_l2, collections=[self.summary_collection], family='loss')
        
        if self.config['reconstruction_loss_type'] == 'l2':
          self.Lx = Lx_end_l2
        elif self.config['reconstruction_loss_type'] == 'binary_cross_entropy':
          self.Lx = Lx_end_bce
        else:
          print('Invalid reconstruction loss type chosen')
          sys.exit()
        
        #############################################################
        # Reconstruction loss - Discriminator
        #############################################################
        if self.df_mode is not None:
          self.Lg = self.discriminator.Lg
          self.d_loss = self.discriminator.loss
        else:
          self.Lg = tf.zeros(1)
          self.d_loss = tf.zeros(1)
        
        #############################################################
        # Latent loss
        #############################################################
        def _latent_loss_loop_body(scaling_factor, t, stop_times, mus, sigmas, logsigmas, KL):
          mu2 = scaling_factor * tf.square(mus.read(t))
          sigma2 = scaling_factor * tf.square(sigmas.read(t))
          logsigma = scaling_factor * logsigmas.read(t)
          kl_term = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - scaling_factor * .5  # each kl term is (1xminibatch)
          KL = tf.add(KL, tf.where(tf.less(t, stop_times), kl_term, tf.zeros(self.batch_size)))
          return [scaling_factor, tf.add(t, 1), stop_times, mus, sigmas, logsigmas, KL]

        scaling_factor = tf.constant(0.1)  # 1.0 / tf.cast(self.T, tf.float32)  # tf.constant(1.0 / self.config['T'])
        t = tf.constant(0)
        KL = tf.zeros([self.batch_size])
        scaling_factor, t, self.stop_times, self.mus, self.sigmas, self.logsigmas, KL = \
          tf.while_loop(lambda s, t, *_: tf.less(t, self.T), _latent_loss_loop_body,
                        [scaling_factor, t, self.stop_times, self.mus, self.sigmas, self.logsigmas, KL],
                        parallel_iterations=1)
        if not self.draw_all_time:
          KL = tf.divide(KL, tf.cast(self.stop_times, dtype=tf.float32))
        self.Lz = tf.reduce_mean(KL)  # average over minibatches
        tf.summary.scalar('Latent Loss', self.Lz, collections=[self.summary_collection], family='loss')
        
        #############################################################
        # Write decision loss
        #############################################################
        def _writing_loss_loop_body(t, stop_times, sw_log_odds, sw_pre_sigmoid, WL):

          def _concrete_binary_kl_mc_sample(
              y, prior_log_odds, prior_temperature, posterior_log_odds, posterior_temperature, eps=10e-10):
            y_times_prior_temp = y * prior_temperature
            log_prior = tf.log(prior_temperature + eps) - y_times_prior_temp + prior_log_odds - \
                2.0 * tf.log(1.0 + tf.exp(-y_times_prior_temp + prior_log_odds) + eps)
            y_times_posterior_temp = y * posterior_temperature
            log_posterior = tf.log(posterior_temperature + eps) - y_times_posterior_temp + posterior_log_odds - \
                2.0 * tf.log(1.0 + tf.exp(-y_times_posterior_temp + posterior_log_odds) + eps)
            return log_posterior - log_prior
          
          kl_term = _concrete_binary_kl_mc_sample(
            sw_pre_sigmoid.read(t), self.write_decision_prior_log_odds, self.write_decision_temperature, sw_log_odds.read(t),
            self.write_decision_temperature) 
          WL = tf.add(WL, tf.where(tf.less(t, stop_times), tf.reshape(kl_term, [-1]), tf.zeros(self.batch_size)))
          return [tf.add(t, 1), stop_times, sw_log_odds, sw_pre_sigmoid, WL]
          
        t = tf.constant(0)
        WL = tf.zeros([self.batch_size])
        t, self.stop_times, self.should_write_log_odds, self.should_write_pre_sigmoid, WL = \
          tf.while_loop(lambda t, *_: tf.less(t, self.T), _writing_loss_loop_body,
                        [t, self.stop_times, self.should_write_log_odds, self.should_write_pre_sigmoid, WL],
                        parallel_iterations=1)
        self.Lwrite = tf.reduce_mean(WL)  # average over minibatches
        tf.summary.scalar('Write Loss', self.Lwrite, collections=[self.summary_collection], family='loss')
        
        #############################################################
        # Movement loss
        #############################################################
        def _movement_loss_loop_body(t, write_bb, total_movement):
          total_movement = tf.add(total_movement, tf.norm(write_bb.read(t + 1)[:, :2] - write_bb.read(t)[:, :2], axis=1))
          return [tf.add(t, 1), write_bb, total_movement]

        total_movement = tf.zeros([self.batch_size])
        t = tf.constant(0)
        t, self.write_bb, total_movement = \
          tf.while_loop(lambda t, a, b: tf.less(t, self.T - 1), _movement_loss_loop_body,
                        [t, self.write_bb, total_movement],
                        parallel_iterations=1)
        total_movement = tf.divide(total_movement, tf.cast(self.stop_times, dtype=tf.float32))
        self.Lmove = tf.reduce_mean(total_movement)
        tf.summary.scalar('Movement Loss', self.Lmove, collections=[self.summary_collection], family='loss')
        
        #############################################################
        # Intensity change loss
        #############################################################
        def _intensity_change_loss_loop_body(t, stop_times, write_bb, total_intensity_change, draw_intensity):
          draw_intensity = draw_intensity.write(
            t, tf.where(tf.less(t, self.stop_times), tf.reshape([write_bb.read(t)[:, 3]], shape=(-1, 1)),
                        tf.zeros((self.batch_size, 1))))
          total_intensity_change = tf.add(total_intensity_change, tf.abs(write_bb.read(t + 1)[:, 3] - write_bb.read(t)[:, 3]))
          return [tf.add(t, 1), stop_times, write_bb, total_intensity_change, draw_intensity]

        total_intensity_change = tf.zeros([self.batch_size])
        t = tf.constant(0)
        t, self.stop_times, self.write_bb, total_intensity_change, self.draw_intensity = \
          tf.while_loop(lambda t, *_: tf.less(t, self.T - 1), _intensity_change_loss_loop_body,
                        [t, self.stop_times, self.write_bb, total_intensity_change, self.draw_intensity],
                        parallel_iterations=1)
        total_intensity_change = tf.divide(total_intensity_change, tf.cast(self.stop_times, dtype=tf.float32))
        self.Lintensity = 100.0 * tf.reduce_mean(total_intensity_change)
        tf.summary.scalar('Intensity Change Loss', self.Lintensity, collections=[self.summary_collection], family='loss')
        summary_intensity = tf.reshape(tf.transpose(self.draw_intensity.stack(), perm=[1, 0, 2]),
                                       shape=(1, self.batch_size, self.config['T'], 1))
        tf.summary.image('Draw Intensity', summary_intensity, max_outputs=1, collections=[self.summary_collection])
        
        #############################################################
        # SSIM loss
        #############################################################
#         bla = (1 - tf.image.ssim(self.input_, self.x_recons, max_val=1.0)) / 2
#         print(bla.shape)
#         Lssim = tf.reduce_mean((1 - tf.image.ssim(self.input_, self.x_recons, max_val=1.0)) / 2)
        
        #############################################################
        # Total loss
        #############################################################
        self.loss = self.Lx + (self.Lg if self.df_mode is not None else 0.0) + self.Lz + (self.Lwrite if not self.draw_all_time else 0.0)
#         self.loss = self.Lx + (self.Lg if self.df_mode is not None else 0.0) + self.Lz + self.Lwrite + self.Lintensity
        tf.summary.scalar('Total Loss', self.loss, collections=[self.summary_collection], family='loss')
        
  def build_optim(self):
    """
    Builds optimizer for training
    """
    if self.mode == 'training':
      # Generator
      optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
      # clip gradients
      grads = optimizer.compute_gradients(self.loss)
      for i, (g, v) in enumerate(grads):

        def _bla(): return True

        def _grad_summary(i, g, v):
          tf.summary.histogram(v.name + '_original_grad', g, collections=[self.summary_collection], family='grads')
          tf.summary.scalar(v.name + '_original_grad_norm', tf.norm(g), collections=[self.summary_collection], family='grads_norm')
          tf.summary.scalar(v.name + '_original_grad_avg', tf.reduce_mean(g), collections=[self.summary_collection], family='grads_avg')
          grads[i] = (tf.clip_by_norm(g, 1.0), v)  # clip gradients
          tf.summary.histogram(v.name + '_clipped_grad', grads[i][0], collections=[self.summary_collection], family='grads')
          tf.summary.scalar(v.name + '_clipped_grad_norm', tf.norm(grads[i][0]), collections=[self.summary_collection], family='grads_norm')
          tf.summary.scalar(v.name + '_clipped_grad_avg', tf.reduce_mean(grads[i][0]), collections=[self.summary_collection], family='grads_avg')
          return True
 
#         tf.cond(tf.greater(tf.count_nonzero(tf.is_nan(g)), 0), lambda: _bla(), lambda: _grad_summary(i, g, v))
      self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
      
  def count_parameters(self):
    """
    Counts the number of trainable parameters in this model
    """
    self.n_parameters = 0
    for v in tf.trainable_variables():
      params = 1
      for s in v.get_shape():
          params *= s.value
      self.n_parameters += params
  
  def get_feed_dict(self, data_batch, canvas_batch):
    """
    Returns the feed dictionary required to run one training step with the model.
    :param data_batch: The mini batch of data to feed into the model
    :param canvas_batch: The mini batch of canvas state at start
    :return: A feed dict that can be passed to a session.run call
    """
    feed_dict = {self.input_: data_batch,
                 self.start_canvas_: canvas_batch}
    return feed_dict
