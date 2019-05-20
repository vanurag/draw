#!/usr/bin/env python

""""
Generates a TFRecord Dataset (train_data.tfrecord) from a set of images

Example Usage: 
  python generate_train_data.py --data_dir=<train_data> --save_dir=<save_path>

Author: Anurag Vempati
"""

import tensorflow as tf
import numpy as np
import sys
import os
import time
from tqdm import tqdm
from config import train_config

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_string("save_dir", "", "")
FLAGS = tf.flags.FLAGS

# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
  """Returns a float_list from an array of float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def load_data(data_dir):
  print('Loading data from {} ...'.format(data_dir))

#   # Reads an image from a file, decodes it into a dense tensor, and resizes it
#   # to a fixed shape.
#   def _parse_function(filename):
#     image_string = tf.read_file(filename)
#     image_gray = tf.image.decode_jpeg(image_string, channels=1)
#     image_converted = tf.image.convert_image_dtype(image_gray, tf.float32)
#     image_resized = tf.image.resize_images(image_converted, [config['A'], config['B']])
#     image_flattened = tf.reshape(image_resized, [-1])
#     return_image = image_flattened
#     if not config['draw_with_white']:
#       return_image = 1.0 - image_flattened
#     return_image = tf.clip_by_value(return_image, 0.0, 0.99)  # for numeric stability during arctanh() operation
#     return return_image

  train_directory = data_dir
  if not os.path.exists(train_directory):
    print("Data not found")
    sys.exit()
  train_files = tf.gfile.ListDirectory(train_directory)
#   train_files = train_files[:10000]
  print('Loading dataset with {} images'.format(len(train_files)))
  idx = np.arange(len(train_files))
  np.random.shuffle(idx)
  train_filenames = [os.path.join(train_directory, train_files[i]) for i in idx]
  
#   train_dataset = tf.data.Dataset.from_tensor_slices(train_filenames)
#   train_dataset = train_dataset.map(_parse_function)
#   train_dataset = train_dataset.repeat().batch(config['batch_size'])
#   train_dataset_iterator = train_dataset.make_one_shot_iterator()
#   next_training_batch = train_dataset_iterator.get_next()
  
  return train_filenames


# Create a dictionary with image features that may be relevant.
def image_example(config, image_string):
#   image_gray = tf.image.decode_jpeg(image_string, channels=1)
#   image_converted = tf.image.convert_image_dtype(image_gray, tf.float32)
#   image_resized = tf.image.resize_images(image_converted, [config['A'], config['B']])
#   image_flattened = tf.reshape(image_resized, [-1])
#   save_image = image_flattened
#   if not config['draw_with_white']:
#     save_image = 1.0 - image_flattened
#   save_image = tf.clip_by_value(save_image, 0.0, 0.99)  # for numeric stability during arctanh() operation

#   before_feature_time = time.time()
  feature = {
      'height': _int64_feature(config['B']),
      'width': _int64_feature(config['A']),
      'depth': _int64_feature(1),
#       'image_raw': _floats_feature(save_image.eval()),
      'image_raw': _bytes_feature(image_string),
  }
#   print('Time to make feature:{} sec'.format(time.time() - before_feature_time))

  return tf.train.Example(features=tf.train.Features(feature=feature))


def main(config):
  
  with tf.Session() as sess:
    # Add the ops to initialize variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # Actually intialize the variables
    sess.run(init_op)
    
    # load the data
    train_filenames = load_data(FLAGS.data_dir)
    
    start_time = time.time()
    with tf.python_io.TFRecordWriter(os.path.join(FLAGS.save_dir, 'train_data.tfrecord')) as writer:
      for fid, filename in tqdm(enumerate(train_filenames)):
#         if (fid + 1) % 1000 == 0 or fid == 0:
#           print('Progress @{} sec: {}/{}'.format(time.time() - start_time, fid + 1, len(train_filenames))) 
        image_string = open(filename, 'rb').read()
#         before_example_time = time.time()
        tf_example = image_example(config, image_string)
#         print('Time to make example:{} sec'.format(time.time() - before_example_time))
        writer.write(tf_example.SerializeToString())
        
    print("Finished writing TFRecord dataset")

  
if __name__ == '__main__':
  main(train_config)
