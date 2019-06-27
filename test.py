#!/usr/bin/env python

""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow

Example Usage: 
  python test.py --data_dir=<test_data>

Author: Anurag Vempati
"""

import tensorflow as tf
import numpy as np
import os
import sys
import math
import time
from config import test_config
from train import get_model_and_placeholders

tf.flags.DEFINE_string("data_dir", "", "")
FLAGS = tf.flags.FLAGS


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
    return_image = image_flattened
    if not config['draw_with_white']:
      return_image = 1.0 - image_flattened
    return_image = tf.clip_by_value(return_image, 0.0, 0.99)  # for numeric stability during arctanh() operation
    return return_image

  test_directory = data_dir
  if not os.path.exists(test_directory):
    print("Test data not found")
    sys.exit()
  test_files = tf.gfile.ListDirectory(test_directory)
  print('Loading dataset with {} images'.format(len(test_files)))
  idx = np.arange(len(test_files))
  np.random.shuffle(idx)
  test_filenames = [os.path.join(test_directory, test_files[i]) for i in idx]
  
  test_dataset = tf.data.Dataset.from_tensor_slices(test_filenames)
  test_dataset = test_dataset.map(_parse_function)
  test_dataset = test_dataset.repeat().batch(config['batch_size'])
  test_dataset_iterator = test_dataset.make_one_shot_iterator()
  next_testing_batch = test_dataset_iterator.get_next()
  
  return len(test_files), next_testing_batch


def main(config):
  if not os.path.exists(config['model_dir']):
    print("Saved model not found")
    sys.exit()
  print('Loading saved model from {}'.format(config['model_dir']))
  
  # load the data
  _, next_data_batch = load_data(config, FLAGS.data_dir)

  # get input placeholders and get the model that we want to train
  draw_model_class, placeholders = get_model_and_placeholders(config)

  # restore the model by first creating the computational graph
  with tf.name_scope('inference'):
    draw_model = draw_model_class(config, placeholders, mode='inference')
    draw_model.build_graph()
    
  with tf.Session() as sess:
    # now restore the trained variables
    # this operation will fail if this `config` does not match the config you used during training
    saver = tf.train.Saver()
    ckpt_id = config['checkpoint_id']
    if ckpt_id is None:
        ckpt_path = os.path.join(os.path.abspath(config['model_dir']), 'drawmodel.ckpt')
    else:
        ckpt_path = os.path.join(os.path.abspath(config['model_dir']), 'drawmodel-{}'.format(ckpt_id))
    print('Evaluating ' + ckpt_path)
    saver.restore(sess, ckpt_path)
    
    # Testing
#     draw_T = config['T']
    xtest = sess.run(next_data_batch)
    cnext = np.zeros([config['batch_size'], config['img_size']])
    
    test_feed_dict = draw_model.get_feed_dict(xtest, cnext)

    # # Logging + Visualization
    log_fetches = {'canvases': draw_model.cs.stack(), 'read_bbs': draw_model.read_bb.stack(), \
                   'write_bbs': draw_model.write_bb.stack(), 'write_times': draw_model.stop_times}
    log_out = sess.run(log_fetches, test_feed_dict)  # generate some examples
    canvases = np.array(log_out['canvases'])  # T x batch x img_size
    read_bounding_boxes = np.array(log_out['read_bbs'])  # T x batch x 3
    write_bounding_boxes = np.array(log_out['write_bbs'])  # T x batch x 4
    write_times = np.array(log_out['write_times'])  # batch
    
    log_file = os.path.join(config['model_dir'], "draw_data.npy")
    np.save(log_file, [xtest, canvases, read_bounding_boxes, write_bounding_boxes, write_times, config['draw_with_white']])
    print("Visualization outputs saved in file: %s" % log_file)
    

if __name__ == '__main__':
    main(test_config)
