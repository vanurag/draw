#!/usr/bin/env python

""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow

Example Usage: 
  python draw.py --data_dir=/tmp/draw --log_dir=/tmp/draw/logs

Author: Eric Jang
"""

import tensorflow as tf
import numpy as np
import os
import sys
import math
import time
from config import train_config
from model import DrawModel

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_string("log_dir", "", "")
FLAGS = tf.flags.FLAGS


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

          
def load_data(config, data_dir):
  print('Loading data from {} ...'.format(data_dir))

  # Reads an image from a file, decodes it into a dense tensor, and resizes it
  # to a fixed shape.
  def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_gray = tf.image.decode_jpeg(image_string, channels=1)
    image_converted = tf.image.convert_image_dtype(image_gray, tf.float32)
    image_resized = tf.image.resize_images(image_converted, [config['A'], config['B']])
    image_flattened = tf.reshape(image_resized, [-1])
    if not config['draw_with_white']:
      image_flipped = 1.0 - image_flattened
      return image_flipped
    return image_flattened

  data_directory = data_dir
  if not os.path.exists(data_directory):
    print("Train data not found")
    sys.exit()
  data_files = tf.gfile.ListDirectory(data_directory)
  print('Loading dataset with {} images'.format(len(data_files)))
  idx = np.arange(len(data_files))
  np.random.shuffle(idx)
  data_filenames = [os.path.join(data_directory, data_files[i]) for i in idx]
  # data_filenames = data_filenames[:1000]  # DEBUG
  dataset = tf.data.Dataset.from_tensor_slices(data_filenames)
  dataset = dataset.map(_parse_function)
  train_dataset = dataset.repeat().batch(config['batch_size'])
  train_dataset_iterator = train_dataset.make_one_shot_iterator()
  next_training_batch = train_dataset_iterator.get_next()
  return len(data_files), next_training_batch


def get_model_and_placeholders(config):
    # create placeholders that we need to feed the required data into the model
    input_pl = tf.placeholder(tf.float32, shape=(config['batch_size'], config['img_size']))
    placeholders = {'input_pl': input_pl}
    return DrawModel, placeholders


def main(config):
  # create unique output directory for this model
  timestamp = str(int(time.time()))
  config['model_dir'] = os.path.abspath(os.path.join(FLAGS.log_dir, 'DRAW' + '_' + timestamp))
  os.makedirs(config['model_dir'])
  print('Logging data to {}'.format(config['model_dir']))
  
  # Export configuration for the current run
  export_config(config, os.path.join(config['model_dir'], 'config.txt'))
  
  # load the data
  _, next_data_batch = load_data(config, FLAGS.data_dir)

  # get input placeholders and get the model that we want to train
  draw_model_class, placeholders = get_model_and_placeholders(config)

  # Create a variable that stores how many training iterations we performed.
  # This is useful for saving/storing the network
  global_step = tf.Variable(1, name='global_step', trainable=False)

  # create a training graph, this is the graph we will use to optimize the parameters
  print('Building training graph')
  with tf.name_scope('training'):
    draw_model = draw_model_class(config, placeholders, mode='training')
    draw_model.build_graph()
    print('created DRAW model with {} parameters'.format(draw_model.n_parameters))
      
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
    
    # Optimizer
    params = tf.trainable_variables()
    print('Building train optimizer')
#         optimizer = tf.train.GradientDescentOptimizer(lr)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
    # clip gradients
    grads, _ = tf.clip_by_global_norm(tf.gradients(draw_model.loss, params), 5)
    train_op = optimizer.apply_gradients(
      zip(grads, params), global_step=global_step)
    print('Finished Building train optimizer')

  print('Building valid graph')
  with tf.name_scope('validation'):
    draw_model_valid = draw_model_class(config, placeholders, mode='validation')
    draw_model_valid.build_graph()
    print('Finished Building valid graphs')
      
  with tf.Session() as sess:
    # Add the ops to initialize variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # Actually intialize the variables
    sess.run(init_op)
    
    # Summaries
    training_summaries = tf.summary.merge(tf.get_collection('training_summaries'))
    train_writer = tf.summary.FileWriter(config['model_dir'] + '/summary/train', sess.graph)
    valid_summaries = tf.summary.merge(tf.get_collection('validation_summaries'))
    valid_writer = tf.summary.FileWriter(config['model_dir'] + '/summary/validation', sess.graph)
    
    # create a saver for writing training checkpoints
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=50)

    # start training
    lowest_test_loss = 1.0e6
    last_saved_epoch = 0  # epoch corresponding to last saved chkpnt
    for i in range(config['train_iters']):
      step = tf.train.global_step(sess, global_step)
      if config['learning_rate_type'] == 'linear' and i % config['learning_rate_decay_steps'] == 0:
        sess.run(lr_decay_op)
        
      xnext = sess.run(next_data_batch)
      
      # Validate every 100th iteration
      if i % 100 == 0:
        valid_feed_dict = draw_model_valid.get_feed_dict(xnext)
        valid_fetches = {'summaries': valid_summaries,
                         'reconstruction_loss': draw_model_valid.Lx,
                         'latent_loss': draw_model_valid.Lz,
                         'loss': draw_model_valid.loss}
        valid_out = sess.run(valid_fetches, valid_feed_dict)
        # For saving plot data
        xlog = xnext
        cost = valid_out['loss']
        print("iter=%d : Lx: %f Lz: %f cost: %f" % (i, valid_out['reconstruction_loss'], valid_out['latent_loss'], cost))
        valid_writer.add_summary(valid_out['summaries'], global_step=step)
        # save this checkpoint if necessary
        if (i - last_saved_epoch + 1) >= config['save_checkpoints_every_epoch'] and cost < lowest_test_loss:
          last_saved_epoch = i
          lowest_test_loss = cost
          saver.save(sess, os.path.join(config['model_dir'], 'drawmodel'), i)
      else:
        train_feed_dict = draw_model.get_feed_dict(xnext)
        train_fetches = {'summaries': training_summaries,
                         'train_op': train_op}
        train_out = sess.run(train_fetches, train_feed_dict)
        if i % 100 == 1:
          train_writer.add_summary(train_out['summaries'], global_step=step)
    
    print('Training finished.')

    # # Logging + Visualization
    log_fetches = {'canvases': draw_model_valid.cs, 'read_bbs': draw_model_valid.read_bb, 'write_bbs': draw_model_valid.write_bb}
    log_out = sess.run(log_fetches, valid_feed_dict)  # generate some examples
    canvases = np.array(log_out['canvases'])  # T x batch x img_size
    read_bounding_boxes = np.array(log_out['read_bbs'])  # T x batch x 3
    write_bounding_boxes = np.array(log_out['write_bbs'])  # T x batch x 3
    
    log_file = os.path.join(config['model_dir'], "draw_data.npy")
    np.save(log_file, [xlog, canvases, read_bounding_boxes, write_bounding_boxes, config['draw_with_white']])
    print("Visualization outputs saved in file: %s" % log_file)
    
    ckpt_file = os.path.join(config['model_dir'], "drawmodel.ckpt")
    print("Model saved in file: %s" % saver.save(sess, ckpt_file))


if __name__ == '__main__':
    main(train_config)
