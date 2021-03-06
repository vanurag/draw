""""
Layer-wise drawing of a given texture

Example Usage: 
  python draw_texture.py --test_dir=<test_data> --draw_width=64 --draw_height=64

Author: Anurag Vempati
"""

import os
import tensorflow as tf
import numpy as np
import sys
from scipy import ndimage
import time

from config import texture_config
from train import get_model_and_placeholders
import matplotlib.pyplot as plt

tf.flags.DEFINE_string("test_dir", "", "")
tf.flags.DEFINE_string("output_dir", "", "")
tf.flags.DEFINE_integer("draw_width", 32, "Width of the draw result")
tf.flags.DEFINE_integer("draw_height", 32, "Height of the draw result")
FLAGS = tf.flags.FLAGS


def load_data(img_width, img_height, flip_image, batch_size, data_dir):
  print('Loading data from {} ...'.format(data_dir))

  # Reads an image from a file, decodes it into a dense tensor, and resizes it
  # to a fixed shape.
  def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_gray = tf.image.decode_jpeg(image_string, channels=1)
    image_converted = tf.image.convert_image_dtype(image_gray, tf.float32)
    image_resized = tf.image.resize_images(image_converted, [img_width, img_height])
    image_flattened = tf.reshape(image_resized, [-1])
    return_image = image_flattened
    if flip_image:  # not config['draw_with_white']:
      return_image = 1.0 - image_flattened
    return_image = tf.clip_by_value(return_image, 0.0, 0.99)  # for numeric stability during arctanh() operation
    return return_image

  train_directory = data_dir
  if not os.path.exists(train_directory):
    print("Train data not found")
    sys.exit()
  train_files = tf.gfile.ListDirectory(train_directory)
#   train_files = train_files[:10000]
  print('Loading dataset with {} images'.format(len(train_files)))
  idx = np.arange(len(train_files))
  np.random.shuffle(idx)
  train_filenames = [os.path.join(train_directory, train_files[i]) for i in idx]
  
  train_dataset = tf.data.Dataset.from_tensor_slices(train_filenames)
  train_dataset = train_dataset.map(_parse_function)
  train_dataset = train_dataset.repeat().batch(batch_size)
  train_dataset_iterator = train_dataset.make_one_shot_iterator()
  next_training_batch = train_dataset_iterator.get_next()
  
  return len(train_files), next_training_batch


def get_next_layer(residual, write_radius):
  # slice
  slice_threshold = 0.2
  M1 = (np.logical_and(residual > 0, residual < slice_threshold)).astype(np.float)
  F1 = M1 * residual
  M2 = (residual >= slice_threshold).astype(np.float)
  F2 = M2 * slice_threshold
  L = F1 + F2
  # filter
  L = ndimage.gaussian_filter(L, sigma=write_radius / 2)
  return L


def export_draw_result_to_file(file_handle, write_bbs):
  for t in range(len(write_bbs)):
    print >> file_handle, "%4f %4f %4f %4f" % (write_bbs[t, 0], write_bbs[t, 1], write_bbs[t, 2], write_bbs[t, 3])
  return

  
def main(config):
  # Load texture image
  n_test_textures, next_texture = load_data(
    FLAGS.draw_width, FLAGS.draw_height, not config['draw_with_white'], config['batch_size'], FLAGS.test_dir)
  
  # Create output file for exporting result
  if not os.path.exists(FLAGS.output_dir):
    print("Output path doesn't exist")
    sys.exit()
  output_file = open(os.path.join(FLAGS.output_dir, 'draw_path.txt'), 'w')
  
  # get input placeholders and get the model that we want to test
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
    
    n_layers = 5
    f, arr = plt.subplots(2 + n_layers, 2 * n_test_textures)
    for t in range(n_test_textures):
      start_time = time.time()
      xnext = sess.run(next_texture)
      input_texture = np.reshape(xnext, (FLAGS.draw_height, FLAGS.draw_width))
      end_result = np.zeros(input_texture.shape)
      
      # Draw layers
      ideal_end_result = np.zeros(input_texture.shape)
      residual = input_texture
      for l in range(n_layers):
        if l == 0:
          # Base coat
          base_coat = np.quantile(input_texture, 0.01)
          M0 = (input_texture >= base_coat).astype(np.float)
          F0 = base_coat * M0
          L_ref = ndimage.gaussian_filter(F0, sigma=config['write_radius'] / 2)
        else:
          # Get next layer
          L_ref = get_next_layer(residual, config['write_radius'])
        ideal_end_result = ideal_end_result + L_ref
        
        # Get patches of config['B'] x config['A'] size to draw
        draw_result = np.zeros((FLAGS.draw_height, FLAGS.draw_width))
        n_patches_a = FLAGS.draw_width / config['A']
        n_patches_b = FLAGS.draw_height / config['B']
        for p_b in range(n_patches_b):
          for p_a in range(n_patches_a):
            patch_row = config['B'] * p_b
            patch_col = config['A'] * p_a
            patch = L_ref[patch_row:patch_row + config['B'], patch_col:patch_col + config['A']]
            xref = np.reshape(patch, (1, config['img_size']))
            cref = np.zeros([config['batch_size'], config['img_size']])
            
            feed_dict = draw_model.get_feed_dict(xref, cref)
            fetches = {'canvases': draw_model.cs.stack(), 'write_bbs': draw_model.write_bb.stack(),
                       'write_times': draw_model.stop_times}
            test_out = sess.run(fetches, feed_dict)
            # results
            canvases = np.concatenate(test_out['canvases'])  # T x img_size
            write_bounding_boxes = np.reshape(np.array(test_out['write_bbs']), (config['T'], 4))  # T x 4
            write_times = test_out['write_times']
            
            # export
            export_draw_result_to_file(output_file, write_bounding_boxes)
            
            draw_result[patch_row:patch_row + config['B'], patch_col:patch_col + config['A']] = \
              np.reshape(canvases[write_times - 1, :], (config['B'], config['A']))
        
        L_ref_viz = 1 - L_ref if not config['draw_with_white'] else L_ref
        draw_result_viz = 1 - draw_result if not config['draw_with_white'] else draw_result
        arr[l + 1, 2 * t].imshow(L_ref_viz, cmap='gray', vmin=0, vmax=1)
        arr[l + 1, 2 * t].set_xticks([])
        arr[l + 1, 2 * t].set_yticks([])
        arr[l + 1, 2 * t + 1].imshow(draw_result_viz, cmap='gray', vmin=0, vmax=1)
        arr[l + 1, 2 * t + 1].set_xticks([])
        arr[l + 1, 2 * t + 1].set_yticks([])
        
        residual = residual - draw_result
        end_result = end_result + draw_result
      
      end_time = time.time()
      print("Time taken to draw texture #%d: %f" % (t + 1, end_time - start_time))
      input_texture_viz = 1 - input_texture if not config['draw_with_white'] else input_texture
      ideal_end_result_viz = 1 - ideal_end_result if not config['draw_with_white'] else ideal_end_result
      end_result_viz = 1 - end_result if not config['draw_with_white'] else end_result
      # Ref texture
      arr[0, 2 * t].imshow(input_texture_viz, cmap='gray', vmin=0, vmax=1)
      arr[0, 2 * t].set_xticks([])
      arr[0, 2 * t].set_yticks([])
      # Ref texture
      arr[0, 2 * t + 1].imshow(input_texture_viz, cmap='gray', vmin=0, vmax=1)
      arr[0, 2 * t + 1].set_xticks([])
      arr[0, 2 * t + 1].set_yticks([])
      # best possible end result
      arr[1 + n_layers, 2 * t].imshow(ideal_end_result_viz, cmap='gray', vmin=0, vmax=1)
      arr[1 + n_layers, 2 * t].set_xticks([])
      arr[1 + n_layers, 2 * t].set_yticks([])
      # End result achieved
      arr[1 + n_layers, 2 * t + 1].imshow(end_result_viz, cmap='gray', vmin=0, vmax=1)
      arr[1 + n_layers, 2 * t + 1].set_xticks([])
      arr[1 + n_layers, 2 * t + 1].set_yticks([])
      
    plt.show()
    output_file.close()


if __name__ == '__main__':
    main(texture_config)
