#!/usr/bin/env python

""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow

Example Usage: 
  python train.py --data_file=<TFRecord_file_path> --log_dir=<log_data_path>

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
from model import DrawModel

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
      "height": tf.FixedLenFeature((), tf.int64, -1),
      "width": tf.FixedLenFeature((), tf.int64, -1),
      "depth": tf.FixedLenFeature((), tf.int64, -1),
      "image_raw": tf.FixedLenFeature((), tf.string, "")
    }
    
    parsed_data = tf.parse_single_example(filename, data_fmt)
    image_gray = tf.image.decode_jpeg(parsed_data["image_raw"], channels=1)
    image_converted = tf.image.convert_image_dtype(image_gray, tf.float32)
    image_resized = tf.image.resize_images(image_converted, [config['A'], config['B']])
    image_flattened = tf.reshape(image_resized, [-1])
    return_image = image_flattened
    if not config['draw_with_white']:
      return_image = 1.0 - image_flattened
    return_image = tf.clip_by_value(return_image, 0.0, 0.99)  # for numeric stability during arctanh() operation
    return return_image

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
    input_pl = tf.placeholder(tf.float32, shape=(config['batch_size'], config['img_size']))
    canvas_pl = tf.placeholder(tf.float32, shape=(config['batch_size'], config['img_size']))
    placeholders = {'input_pl': input_pl,
                    'canvas_pl': canvas_pl}
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
  next_data_batch = load_data(config, FLAGS.data_file)

  # get input placeholders and get the model that we want to train
  draw_model_class, placeholders = get_model_and_placeholders(config)
  
  # create a training graph, this is the graph we will use to optimize the parameters
  print('Building training graph')
  with tf.name_scope('training'):
    draw_model = draw_model_class(config, placeholders, mode='training', annealing_schedules=config['annealing_schedules'])
    draw_model.build_graph()
    print('created DRAW model with {} parameters'.format(draw_model.n_parameters))
      
  print('Building valid graph')
  with tf.name_scope('validation'):
    draw_model_valid = draw_model_class(config, placeholders, mode='validation', annealing_schedules=config['annealing_schedules'])
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
    draw_T = config['T']
#     lowest_test_loss = 1.0e6
    last_saved_epoch = 0  # epoch corresponding to last saved chkpnt
    iteration = 0
    previous_epoch = 0
    epoch = 0
    with tqdm() as pbar:
      while epoch < config['n_epochs']:
        # Next data batch
        previous_epoch = epoch
        xnext, epoch = sess.run(next_data_batch)
        epoch = epoch[0]
        if (previous_epoch > 0 and epoch == 0): break
        
        step = tf.train.global_step(sess, draw_model.global_step)
  
        # Hot start
        if config['use_hot_start']:
          crop_fraction = (epoch + 1) * config['crop_fraction_increase_rate']
          if crop_fraction >= 1.0:
            cnext = np.zeros([config['batch_size'], config['img_size']])
            draw_T = config['T']
          else:
            xnext_reshaped = np.copy(xnext)
            xnext_reshaped = xnext_reshaped.reshape((config['batch_size'], config['B'], config['A']))
            start_row = np.random.randint(config['B'] * (1 - crop_fraction))  # , size=config['batch_size'])
            start_col = np.random.randint(config['A'] * (1 - crop_fraction))  # , size=config['batch_size'])
            xnext_reshaped[:, start_row:start_row + int(crop_fraction * config['B']), \
                           start_col:start_col + int(crop_fraction * config['A'])] = 0.0
            cnext = np.reshape(xnext_reshaped, (config['batch_size'], config['img_size']))
            draw_T = max(1, int(config['T'] * crop_fraction))
        else:
          cnext = np.zeros([config['batch_size'], config['img_size']])
          draw_T = config['T']
          
        # Validate every 100th iteration
        if iteration % 100 == 0:
          valid_feed_dict = draw_model_valid.get_feed_dict(xnext, cnext)
          valid_feed_dict[draw_model_valid.T] = draw_T
          valid_feed_dict[draw_model_valid.global_step] = step
          valid_fetches = {'summaries': valid_summaries,
                           'reconstruction_loss': draw_model_valid.Lx,
                           'latent_loss': draw_model_valid.Lz,
                           'write_loss': draw_model_valid.Lwrite,
                           'intensity_change_loss': draw_model_valid.Lintensity,
                           'movement_loss': draw_model_valid.Lmove,
                           'loss': draw_model_valid.loss}
          valid_out = sess.run(valid_fetches, valid_feed_dict)
          # For saving plot data
          xlog = xnext
          cost = valid_out['loss']
          print("epoch=%d, iter=%d : Lx: %f Lz: %f Lwrite: %f cost: %f" % \
                (epoch, iteration, valid_out['reconstruction_loss'], valid_out['latent_loss'], valid_out['write_loss'], cost))
          valid_writer.add_summary(valid_out['summaries'], global_step=step)
          # save this checkpoint if necessary
          if (epoch - last_saved_epoch + 1) >= config['save_checkpoints_every_epoch']:  # and cost < lowest_test_loss:
            last_saved_epoch = epoch
  #           lowest_test_loss = cost
            saver.save(sess, os.path.join(config['model_dir'], 'drawmodel'), epoch)
        else:
          train_feed_dict = draw_model.get_feed_dict(xnext, cnext)
          train_feed_dict[draw_model.T] = draw_T
          train_fetches = {'summaries': training_summaries,
                           'train_op': draw_model.train_op}
          train_out = sess.run(train_fetches, train_feed_dict)
          if iteration % 100 == 1:
            train_writer.add_summary(train_out['summaries'], global_step=step)
            
        iteration += 1
        pbar.update(1)
    
    print('Training finished.')

    # # Logging + Visualization
    log_fetches = {'canvases': draw_model_valid.cs.stack(), 'read_bbs': draw_model_valid.read_bb.stack(), \
                   'write_bbs': draw_model_valid.write_bb.stack(), 'write_times': draw_model_valid.stop_times}
    log_out = sess.run(log_fetches, valid_feed_dict)  # generate some examples
    canvases = np.array(log_out['canvases'])  # T x batch x img_size
    read_bounding_boxes = np.array(log_out['read_bbs'])  # T x batch x 3
    write_bounding_boxes = np.array(log_out['write_bbs'])  # T x batch x 3
    write_times = np.array(log_out['write_times'])  # batch
    
    log_file = os.path.join(config['model_dir'], "draw_data.npy")
    np.save(log_file, [xlog, canvases, read_bounding_boxes, write_bounding_boxes, write_times, config['draw_with_white']])
    print("Visualization outputs saved in file: %s" % log_file)
    
    ckpt_file = os.path.join(config['model_dir'], "drawmodel.ckpt")
    print("Model saved in file: %s" % saver.save(sess, ckpt_file))


if __name__ == '__main__':
    main(train_config)
