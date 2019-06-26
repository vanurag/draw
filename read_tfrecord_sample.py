#!/usr/bin/env python

import tensorflow as tf

tf.flags.DEFINE_string("file", "", "")
FLAGS = tf.flags.FLAGS

record_iterator = tf.python_io.tf_record_iterator(path=FLAGS.file)

record_size = sum(1 for _ in record_iterator)
print('record size: {}'.format(record_size))

for string_record in record_iterator:
  example = tf.train.Example()
  example.ParseFromString(string_record)
  
  print(example)
  
  # Exit after 1 iteration since we just display a sample
  break
