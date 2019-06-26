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


class Discriminator(object):
  """
  Tells whether reference and query images look indistinguishable.
  Creates training and validation computational graphs.
  Note that tf.variable_scope enables parameter sharing so that both graphs are identical.
  """
  
  def __init__(self, config, placeholders, mode, annealing_schedules=None):
    """
    Basic setup.
    :param config: configuration dictionary
    :param placeholders: dictionary of input placeholders
    :param mode: training, validation or inference
    """
    assert mode in ['training', 'validation', 'inference']
    self.config = config
    self.real_input_ = placeholders['real_input_pl']
    self.fake_input_ = placeholders['fake_input_pl']
    self.mode = mode
    self.DO_SHARE = True if self.mode == 'validation' else None
    self.A = config['A']  # image width
    self.B = config['B']  # image height
    self.img_size = config['img_size']  # the canvas size
    self.draw_with_white = config['draw_with_white']  # draw with white ink or black ink
    
    # Discriminator
    self.df_mode = config['disc_mode']  # dcgan, wgan, wgan-gp, or None
    if self.df_mode is None:
      print('Choose an appropriate Discriminator mode!')
      sys.exit()
#     self.y_dim = None
    self.df_dim = 64  # num filters in first conv layer
#     self.dfc_dim = 1024  # fully connected layer units
    # batch normalization : deals with poor initialization helps gradient flow
#     self.d_bn = False
#     self.d_bn1 = batch_norm(name='d_bn1')
#     self.d_bn2 = batch_norm(name='d_bn2')
#     self.d_bn3 = batch_norm(name='d_bn3')
#     self.d_grad_penalty = False  # whether to use WGAN with gradient penalty
    
    self.batch_size = config['batch_size']  # training minibatch size
    self.n_summary_per_batch = config['n_summary_per_batch'] 

    self.summary_collection = 'training_summaries' if mode == 'training' else 'validation_summaries'
    
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
    
  def LeakyReLU(self, x, alpha=0.2):
    return tf.maximum(alpha * x, x)
  
  def build_summary(self, input):
    if self.draw_with_white:
      s_reference = tf.transpose(\
        tf.reshape(input[:self.n_summary_per_batch, :self.img_size], \
                   [self.n_summary_per_batch, self.B, self.A]), perm=[1, 0, 2])
      s_query = tf.transpose(\
        tf.reshape(input[:self.n_summary_per_batch, self.img_size:], \
                   [self.n_summary_per_batch, self.B, self.A]), perm=[1, 0, 2])
    else:
      s_reference = tf.transpose(\
        tf.reshape(1 - input[:self.n_summary_per_batch, :self.img_size], \
                   [self.n_summary_per_batch, self.B, self.A]), perm=[1, 0, 2])
      s_query = tf.transpose(\
        tf.reshape(1 - input[:self.n_summary_per_batch, self.img_size:], \
                   [self.n_summary_per_batch, self.B, self.A]), perm=[1, 0, 2])
      
    return s_reference, s_query
    
  def build_graph(self):
    s_reference_real, s_query_real = self.build_summary(self.real_input_)
    tf.summary.image('reference_real', \
                     tf.reshape(s_reference_real, [1, self.B, self.n_summary_per_batch * self.A, 1]), max_outputs=1, \
                     collections=[self.summary_collection])  # [1, B, A, 1]
    tf.summary.image('query_real', \
                     tf.reshape(s_query_real, [1, self.B, self.n_summary_per_batch * self.A, 1]), max_outputs=1, \
                     collections=[self.summary_collection])  # [1, B, A, 1]
    s_reference_fake, s_query_fake = self.build_summary(self.fake_input_)
    tf.summary.image('reference_fake', \
                     tf.reshape(s_reference_fake, [1, self.B, self.n_summary_per_batch * self.A, 1]), max_outputs=1, \
                     collections=[self.summary_collection])  # [1, B, A, 1]
    tf.summary.image('query_fake', \
                     tf.reshape(s_query_fake, [1, self.B, self.n_summary_per_batch * self.A, 1]), max_outputs=1, \
                     collections=[self.summary_collection])  # [1, B, A, 1]
      
    self.real_probs, self.real_logits = self.build_model(self.real_input_)
    tf.summary.histogram('Discriminator result on input', self.real_probs, collections=[self.summary_collection])
    self.DO_SHARE = True
    self.fake_probs, self.fake_logits = self.build_model(self.fake_input_)
    tf.summary.histogram('Discriminator result on reconstruction', self.fake_probs, collections=[self.summary_collection])
    self.build_loss()
    self.build_optim()
    self.count_parameters()
  
  def build_model(self, input):
    """
    Builds the actual model.
    """
    with tf.variable_scope("draw_discriminator", reuse=self.DO_SHARE):
      disc_logits = tf.reshape(input, [-1, 1, 2 * self.B, self.A])
  
      disc_logits = lib.ops.conv2d.Conv2D('Discriminator.1', 1, self.df_dim, 5, disc_logits, stride=2)
      disc_logits = self.LeakyReLU(disc_logits)
  
      disc_logits = lib.ops.conv2d.Conv2D('Discriminator.2', self.df_dim, 2 * self.df_dim, 5, disc_logits, stride=2)
      if self.df_mode == 'wgan':
          disc_logits = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 2, 3], disc_logits)
      disc_logits = self.LeakyReLU(disc_logits)
  
      disc_logits = lib.ops.conv2d.Conv2D('Discriminator.3', 2 * self.df_dim, 4 * self.df_dim, 5, disc_logits, stride=2)
      if self.df_mode == 'wgan':
          disc_logits = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 2, 3], disc_logits)
      disc_logits = self.LeakyReLU(disc_logits)
  
      disc_logits = tf.reshape(disc_logits, [-1, 4 * 4 * 4 * self.df_dim])
      disc_logits = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 4 * self.df_dim, 1, disc_logits)
      
      disc_logits = tf.reshape(disc_logits, [-1])
      disc_probs = tf.nn.sigmoid(disc_logits)
      
      return disc_probs, disc_logits
  
  def build_loss(self):
    """
    Builds the loss function.
    """
    # only need loss if we are not in inference mode
    if self.mode is not 'inference':
      with tf.name_scope('loss'):

        if self.df_mode is not None:
          if self.df_mode == 'wgan':
            self.loss = tf.reduce_mean(self.fake_logits) - tf.reduce_mean(self.real_logits)
        
          elif self.df_mode == 'wgan-gp':
            critic_loss = tf.reduce_mean(self.fake_logits) - tf.reduce_mean(self.real_logits)
            tf.summary.scalar('Critic Loss', critic_loss, collections=[self.summary_collection], family='loss')
        
            alpha = tf.random_uniform(
              shape=[self.batch_size, 1],
              minval=0.,
              maxval=1.
            )
            differences = self.fake_input_ - self.real_input_
            interpolates = self.real_input_ + (alpha * differences)
            gradients = tf.gradients(self.discriminator(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            tf.summary.scalar('Gradient penalty', gradient_penalty, collections=[self.summary_collection], family='loss')
            self.loss = critic_loss + 10.0 * gradient_penalty
        
          elif self.df_mode == 'dcgan':
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.fake_logits,
              labels=tf.zeros_like(self.fake_logits)
            ))
            tf.summary.scalar('Discriminator Loss (fake)', d_loss_fake, collections=[self.summary_collection], family='loss')
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.real_logits,
              labels=tf.ones_like(self.real_logits)
            ))
            tf.summary.scalar('Discriminator Loss (real)', d_loss_real, collections=[self.summary_collection], family='loss')
            self.loss = (d_loss_real + d_loss_fake) / 2.
          
          tf.summary.scalar('Discriminator Loss', self.loss, collections=[self.summary_collection], family='loss')
        
        tf.summary.scalar('Total Loss', self.loss, collections=[self.summary_collection], family='loss')
        
  def build_optim(self):
    """
    Builds optimizer for training
    """
    if self.mode == 'training':
      # Discriminator
      if self.df_mode is not None:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
  
        # clip weights
        clip_ops = []
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'):
          clip_bounds = [-.01, .01]
          clip_ops.append(
            tf.assign(
              var,
              tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
              )
            )
        self.clip_disc_weights = tf.group(*clip_ops)
      
        grads = optimizer.compute_gradients(self.loss)
  #       for i, (g, v) in enumerate(d_grads):
  #         if g is not None:
  #           grads[i] = (tf.clip_by_norm(g, 1.0), v)  # clip gradients
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
  
  def get_feed_dict(self, real_data_batch, fake_data_batch):
    """
    Returns the feed dictionary required to run one training step with the model.
    :param real_data_batch: The mini batch of real data to feed into the model
    :param fake_data_batch: The mini batch of fake data to feed into the model
    :return: A feed dict that can be passed to a session.run call
    """
    feed_dict = {self.real_input_: real_data_batch,
                 self.fake_input_: fake_data_batch}
    return feed_dict
