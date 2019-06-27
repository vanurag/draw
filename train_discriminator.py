#!/usr/bin/env python

""""
Train a discriminator for matching real textures against machine drawn results

Example Usage: 
  python train_discriminator.py --data_file=<TFRecord_file_path> --log_dir=<log_data_path>

Author: Anurag Vempati
"""

import tensorflow as tf
import numpy as np
import os
import sys
import math
import time
from tqdm import tqdm
from config import train_config
from discriminator import Discriminator

tf.flags.DEFINE_string("data_file", "", "")
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

          
def load_data(config, data_file):
  print('Loading data from {} ...'.format(data_file))

  # Reads an image from a file, decodes it into a dense tensor, and resizes it
  # to a fixed shape.
  def _parse_function(filename):
    data_fmt = {
      "real": tf.FixedLenFeature([config['img_size']], tf.float32, tf.zeros(config['img_size'])),
      "fake": tf.FixedLenFeature([config['img_size']], tf.float32, tf.zeros(config['img_size']))
    }
    
    parsed_data = tf.parse_single_example(filename, data_fmt)
    real_image = tf.squeeze(parsed_data["real"])  # I
    fake_image = tf.squeeze(parsed_data["fake"])  # I
    
    real_data = tf.concat([real_image, real_image], 0)  # (real, real) # 2*I
    fake_data = tf.concat([real_image, fake_image], 0)  # (real, fake) # 2*I
    
    return [real_data, fake_data]
    
  if not os.path.exists(data_file):
    print("Data TFRecord not found")
    sys.exit()
  
  data_files = tf.data.Dataset.list_files(data_file)
  dataset = data_files.interleave(tf.data.TFRecordDataset, cycle_length=2)
  dataset = dataset.map(map_func=_parse_function, num_parallel_calls=4)
  epoch_counter = tf.data.TFRecordDataset.range(config['n_epochs'])
  dataset = epoch_counter.flat_map(lambda i: tf.data.Dataset.zip(
    (dataset, tf.data.Dataset.from_tensors(i).repeat())))
  dataset = dataset.repeat()
  dataset = dataset.batch(config['batch_size'])
  dataset = dataset.prefetch(buffer_size=config['batch_size'])
  dataset_iterator = dataset.make_one_shot_iterator()
  next_data_batch = dataset_iterator.get_next()
  
  return next_data_batch


def get_model_and_placeholders(config):
    # create placeholders that we need to feed the required data into the model
    real_data_pl = tf.placeholder(tf.float32, shape=(config['batch_size'], 2 * config['img_size']))
    fake_data_pl = tf.placeholder(tf.float32, shape=(config['batch_size'], 2 * config['img_size']))
    placeholders = {'real_input_pl': real_data_pl,
                    'fake_input_pl': fake_data_pl}
    return Discriminator, placeholders


def main(config):
  # create unique output directory for this model
  timestamp = str(int(time.time()))
  config['model_dir'] = os.path.abspath(os.path.join(FLAGS.log_dir, 'DISC' + '_' + timestamp))
  os.makedirs(config['model_dir'])
  print('Logging data to {}'.format(config['model_dir']))
  
  # Export configuration for the current run
  export_config(config, os.path.join(config['model_dir'], 'config.txt'))
  
  # load the data
  next_data_batch = load_data(config, FLAGS.data_file)

  # get input placeholders and get the model that we want to train
  disc_model_class, placeholders = get_model_and_placeholders(config)
  
  # create a training graph, this is the graph we will use to optimize the parameters
  print('Building training graph')
#   with tf.name_scope('training'):
  disc_model = disc_model_class(config, placeholders, mode='training', annealing_schedules=None)
  disc_model.build_graph()
  print('created Discriminator with {} parameters'.format(disc_model.n_parameters))
      
  print('Building valid graph')
#   with tf.name_scope('validation'):
  disc_model_valid = disc_model_class(config, placeholders, mode='validation', annealing_schedules=None)
  disc_model_valid.build_graph()
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
    last_saved_epoch = 0  # epoch corresponding to last saved chkpnt
    iteration = 0
    previous_epoch = 0
    epoch = 0
    with tqdm() as pbar:
      while epoch < config['n_epochs']:
        # Next data batch
        previous_epoch = epoch
#         bla = sess.run(next_data_batch)
#         print('size: ', len(bla))
        data_next, epoch = sess.run(next_data_batch)
        real_next = data_next[0]
        fake_next = data_next[1]
        epoch = epoch[0]
        if (previous_epoch > 0 and epoch == 0): break
        
        step = tf.train.global_step(sess, disc_model.global_step)
        
        # Validate every 100th iteration
        if iteration % 100 == 0:
          valid_feed_dict = disc_model_valid.get_feed_dict(real_next, fake_next)
          valid_feed_dict[disc_model_valid.global_step] = step
          valid_fetches = {'loss': disc_model_valid.loss, 'summaries': valid_summaries}
          valid_out = sess.run(valid_fetches, valid_feed_dict)
          
          cost = valid_out['loss']
          print("epoch=%d, iter=%d : cost: %f" % (epoch, iteration, valid_out['loss']))
          valid_writer.add_summary(valid_out['summaries'], global_step=step)
          # save this checkpoint if necessary
          if (epoch - last_saved_epoch + 1) >= config['save_checkpoints_every_epoch']:  # and cost < lowest_test_loss:
            last_saved_epoch = epoch
  #           lowest_test_loss = cost
            saver.save(sess, os.path.join(config['model_dir'], 'discmodel'), epoch)
        else:
          # Train discriminator
          train_feed_dict = disc_model.get_feed_dict(real_next, fake_next)
          train_fetches = {'train_op': disc_model.train_op, 'summaries': training_summaries}
          train_out = sess.run(train_fetches, train_feed_dict)
          if not disc_model.df_mode == 'wgan':
            _ = sess.run(disc_model.clip_disc_weights)
          if iteration % 100 == 1:
            train_writer.add_summary(train_out['summaries'], global_step=step)
            
        iteration += 1
        pbar.update(1)
    
    print('Training finished.')

    ckpt_file = os.path.join(config['model_dir'], "discmodel.ckpt")
    print("Model saved in file: %s" % saver.save(sess, ckpt_file))


if __name__ == '__main__':
    main(train_config)
