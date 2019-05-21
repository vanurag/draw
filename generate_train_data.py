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
from tqdm import tqdm

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
  
  return train_filenames


# Create a dictionary with image features that may be relevant.
def image_example(image_string):

  feature = {
    'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))


def main():
  
  with tf.Session() as sess:
    # Add the ops to initialize variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # Actually intialize the variables
    sess.run(init_op)
    
    # load the data
    train_filenames = load_data(FLAGS.data_dir)
    
    with tf.python_io.TFRecordWriter(os.path.join(FLAGS.save_dir, 'train_data.tfrecord')) as writer:
      for _, filename in tqdm(enumerate(train_filenames)):
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string)
        writer.write(tf_example.SerializeToString())
        
    print("Finished writing TFRecord dataset")

  
if __name__ == '__main__':
  main()
