import tensorflow as tf
import math


class DrawModel(object):
  """
  Creates training and validation computational graphs.
  Note that tf.variable_scope enables parameter sharing so that both graphs are identical.
  """
  
  def __init__(self, config, placeholders, mode):
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
    
    self.batch_size = config['batch_size']  # training minibatch size
    self.n_summary_per_batch = config['n_summary_per_batch'] 
#     self.train_iters = config['train_iters'] 
#     self.save_checkpoints_every_epoch = config['save_checkpoints_every_epoch']  # save chpnt after atleast these many epochs
#     self.learning_rate = config['learning_rate']  # learning rate for optimizer
#     self.learning_rate_type = config['learning_rate_type']  # ['fixed', 'exponential', 'linear']
#     self.learning_rate_decay_steps = config['learning_rate_decay_steps']
#     self.learning_rate_decay_rate = config['learning_rate_decay_rate']
    self.eps = config['eps']  # epsilon for numerical stability

    self.summary_collection = 'training_summaries' if mode == 'training' else 'validation_summaries'
    
    self.e = tf.random_normal((config['batch_size'], config['z_size']), mean=0, stddev=1)  # Qsampler noise
    
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
    self.cs = tf.TensorArray(dtype=tf.float32, size=self.T,
                             dynamic_size=False,
                             element_shape=tf.TensorShape([self.batch_size, self.img_size]),
                             clear_after_read=False)
    # summary of canvases B x batch_size x A
    self.scs = tf.TensorArray(dtype=tf.float32, size=self.T,
                              dynamic_size=False,
                              element_shape=tf.TensorShape([self.B, self.n_summary_per_batch, self.A]))
#     self.read_bb = [0] * self.T  # sequence of bounding boxes for reading (center (x,y), (read_n-1)*delta + 4*sigma)
    self.read_bb = tf.TensorArray(dtype=tf.float32, size=self.T,
                                  dynamic_size=False,
                                  element_shape=tf.TensorShape([self.batch_size, 3]))
#     self.write_bb = [0] * self.T  # sequence of bounding boxes for writing (center (x,y), 4*sigma)
    self.write_bb = tf.TensorArray(dtype=tf.float32, size=self.T,
                                   dynamic_size=False,
                                   element_shape=tf.TensorShape([self.batch_size, 3]),
                                   clear_after_read=False)
#     self.mus, self.logsigmas, self.sigmas = [0] * self.T, [0] * self.T, [0] * self.T  # gaussian params generated by SampleQ. We will need these for computing loss.
    self.mus = tf.TensorArray(dtype=tf.float32, size=self.T,
                              dynamic_size=False,
                              element_shape=tf.TensorShape([self.batch_size, self.z_size]))
    self.logsigmas = tf.TensorArray(dtype=tf.float32, size=self.T,
                                    dynamic_size=False,
                                    element_shape=tf.TensorShape([self.batch_size, self.z_size]))
    self.sigmas = tf.TensorArray(dtype=tf.float32, size=self.T,
                                 dynamic_size=False,
                                 element_shape=tf.TensorShape([self.batch_size, self.z_size]))
      
    self.read = self.read_attn if self.use_read_attn else self.read_no_attn
    self.write = self.write_attn if self.use_write_attn else self.write_no_attn
    
  def build_graph(self):
    self.build_model()
    self.build_loss()
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
  
  def linear(self, x, output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w = tf.get_variable("w", [x.get_shape()[1], output_dim])  # , initializer=tf.random_normal_initializer()) 
  #   b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    b = tf.get_variable("b", [output_dim], initializer=tf.random_normal_initializer())
    return tf.matmul(x, w) + b
  
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
      params = self.linear(h_dec, 5)
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
        params = self.linear(h_dec, 5)
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
        mu = self.linear(h_enc, self.z_size)
    with tf.variable_scope("sigma", reuse=self.DO_SHARE):
        logsigma = self.linear(h_enc, self.z_size)
        sigma = tf.exp(logsigma)
    return (mu + sigma * self.e, mu, logsigma, sigma)
  
  # # DECODER ## 
  def decode(self, state, input):
    with tf.variable_scope("decoder", reuse=self.DO_SHARE):
        return self.lstm_dec(input, state)
  
  # # WRITER ## 
  def write_no_attn(self, h_dec):
    with tf.variable_scope("write", reuse=self.DO_SHARE):
        return self.linear(h_dec, self.img_size), tf.concat([0, 0, 0], 1)
  
  def write_attn(self, h_dec):
    with tf.variable_scope("writeW", reuse=self.DO_SHARE):
        w = tf.sigmoid(self.linear(h_dec, self.write_size))  # batch x (write_n*write_n)
  #   w = tf.ones((batch_size, write_size))
    N = self.write_n
    w = tf.reshape(w, [self.batch_size, N, N])
    Fx, Fy, gamma, gx, gy, sigma2, _ = self.write_attn_window("write", h_dec, self.write_n)
    Fyt = tf.transpose(Fy, perm=[0, 2, 1])
    wr = tf.matmul(Fyt, tf.matmul(w, Fx))
    wr = tf.reshape(wr, [self.batch_size, self.B * self.A])
    # gamma=tf.tile(gamma,[1,B*A])
    return wr * tf.reshape(1.0 / gamma, [-1, 1]), tf.concat([gx, gy, 4.0 * tf.sqrt(sigma2)], 1)
  
  def draw_loop_body(self, t, cs, scs, read_bb, write_bb, mus, logsigmas, sigmas, h_dec_prev, enc_state, dec_state):
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
    cs = cs.write(t, c_prev + write_output[0])  # store results
    if self.draw_with_white:
      scs = scs.write(t, tf.transpose(\
        tf.reshape(cs.read(t)[:self.n_summary_per_batch, :], \
                   [self.n_summary_per_batch, self.B, self.A]), perm=[1, 0, 2]))  # B x batch_size x A
    else:
      scs = scs.write(t, tf.transpose(\
        tf.reshape(1 - cs.read(t)[:self.n_summary_per_batch, :], \
                   [self.n_summary_per_batch, self.B, self.A]), perm=[1, 0, 2]))  # B x batch_size x A
    write_bb = write_bb.write(t, write_output[1])
    h_dec_prev = h_dec
    self.DO_SHARE = True  # from now on, share variables
    return [tf.add(t, 1), cs, scs, read_bb, write_bb, mus, logsigmas, sigmas, h_dec_prev, enc_state, dec_state]
    
  def build_model(self):
    """
    Builds the actual model.
    """
    with tf.variable_scope('draw_model'):  # , reuse=self.DO_SHARE):
      # initial states
      h_dec_prev = tf.zeros((self.batch_size, self.dec_size))
      enc_state = self.lstm_enc.zero_state(self.batch_size, tf.float32)
      dec_state = self.lstm_dec.zero_state(self.batch_size, tf.float32)
      
      t = tf.constant(0);
      t, self.cs, self.scs, self.read_bb, self.write_bb, self.mus, self.logsigmas, self.sigmas, h_dec_prev, enc_state, dec_state = \
        tf.while_loop(lambda t, a, b, c, d, e, f, g, h, i, j: tf.less(t, self.T), self.draw_loop_body,
                      [t, self.cs, self.scs, self.read_bb, self.write_bb, self.mus, self.logsigmas, self.sigmas, \
                       h_dec_prev, enc_state, dec_state],
                      parallel_iterations=1)
        
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
        # Reconstruction loss
        anchor_point = tf.cast(self.T / 2, tf.int32)
        # reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
        x_recons = tf.nn.tanh(self.cs.read(self.T - 1))
        x_recons_anchor = tf.nn.tanh(self.cs.read(anchor_point))
        
        # after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
        # reconstruction term
        Lx_end = tf.reduce_sum(self.binary_crossentropy(self.input_, x_recons), 1)
#         Lx_end = tf.norm(self.input_ - x_recons, axis=1)
        Lx_end = tf.reduce_mean(Lx_end)
        # Lx_anchor = tf.reduce_sum(binary_crossentropy(tf.multiply(x, x_recons_anchor), x_recons_anchor), 1)  # reconstruction term
        # reconstruction term
        Lx_anchor = tf.reduce_sum(self.binary_crossentropy(self.input_, x_recons_anchor), 1)  
        Lx_anchor = tf.reduce_mean(Lx_anchor)
        self.Lx = Lx_end  # + 0.2 * Lx_anchor
        tf.summary.scalar('Reconstruction Loss at anchor', Lx_anchor, collections=[self.summary_collection])
        tf.summary.scalar('Reconstruction Loss at end', Lx_end, collections=[self.summary_collection])
        tf.summary.scalar('Total Reconstruction Loss', self.Lx, collections=[self.summary_collection])
        
        # Latent loss
        def _latent_loss_loop_body(scaling_factor, t, mus, sigmas, logsigmas, KL):
          mu2 = scaling_factor * tf.square(mus.read(t))
          sigma2 = scaling_factor * tf.square(sigmas.read(t))
          logsigma = scaling_factor * logsigmas.read(t)
          kl_term = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - scaling_factor * .5  # each kl term is (1xminibatch)
          return [scaling_factor, tf.add(t, 1), mus, sigmas, logsigmas, tf.add(KL, kl_term)]

        scaling_factor = 1.0 / tf.cast(self.T, tf.float32)  # tf.constant(1.0 / self.config['T'])
        t = tf.constant(0)
        KL = tf.zeros([self.batch_size])
        scaling_factor, t, self.mus, self.sigmas, self.logsigmas, KL = \
          tf.while_loop(lambda s, t, a, b, c, d: tf.less(t, self.T), _latent_loss_loop_body,
                        [scaling_factor, t, self.mus, self.sigmas, self.logsigmas, KL],
                        parallel_iterations=1)
        self.Lz = tf.reduce_mean(KL)  # average over minibatches
        tf.summary.scalar('Latent Loss', self.Lz, collections=[self.summary_collection])
        
        # Movement loss
        def _movement_loss_loop_body(t, write_bb, total_movement):
          total_movement = tf.add(total_movement, tf.norm(write_bb.read(t + 1)[:, :2] - write_bb.read(t)[:, :2], axis=1))
          return [tf.add(t, 1), write_bb, total_movement]

        total_movement = tf.zeros([self.batch_size])
        t = tf.constant(0)
        t, self.write_bb, total_movement = \
          tf.while_loop(lambda t, a, b: tf.less(t, self.T - 1), _movement_loss_loop_body,
                        [t, self.write_bb, total_movement],
                        parallel_iterations=1)
        self.Lmove = tf.reduce_mean(total_movement)
        tf.summary.scalar('Movement Loss', self.Lmove, collections=[self.summary_collection])
        
        # SSIM loss
#         bla = (1 - tf.image.ssim(self.input_, x_recons, max_val=1.0)) / 2
#         print(bla.shape)
#         Lssim = tf.reduce_mean((1 - tf.image.ssim(self.input_, x_recons, max_val=1.0)) / 2)
        
        # Total loss
        self.loss = self.Lx + self.Lz
#         self.loss = self.Lx + self.Lz + 0.1 * self.Lmove
#         self.loss = self.Lx + self.Lmove
        tf.summary.scalar('Total Loss', self.loss, collections=[self.summary_collection])
  
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
