import os
import tensorflow as tf
import numpy as np

from config import test_config
from train import load_data, get_model_and_placeholders
import matplotlib.pyplot as plt

tf.flags.DEFINE_string("test_dir", "", "")
FLAGS = tf.flags.FLAGS


def main(config):
  # Load texture image
  n_test_textures, next_texture = load_data(config, FLAGS.test_dir)
  
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
        ckpt_path = tf.train.latest_checkpoint(config['model_dir'])
    else:
        ckpt_path = os.path.join(os.path.abspath(config['model_dir']), 'model-{}'.format(ckpt_id))
    print('Evaluating ' + ckpt_path)
    saver.restore(sess, ckpt_path)
    
    f, arr = plt.subplots(2, n_test_textures + 1 if n_test_textures == 1 else n_test_textures)
    for t in range(n_test_textures):
      xnext = sess.run(next_texture)
      feed_dict = draw_model.get_feed_dict(xnext)
      fetches = {'canvases': draw_model.cs, 'read_bbs': draw_model.read_bb, 'write_bbs': draw_model.write_bb}
      test_out = sess.run(fetches, feed_dict)
      # results
      canvases = np.concatenate(test_out['canvases'])  # T x img_size
      read_bounding_boxes = np.concatenate(test_out['read_bbs'])  # T x 3
      write_bounding_boxes = np.concatenate(test_out['write_bbs'])  # T x 3
      
      if config['draw_with_white']:
        end_result = np.reshape(canvases[-1, :], (config['B'], config['A']))
        input_image = np.reshape(xnext, (config['B'], config['A']))
      else:
        end_result = np.reshape(1 - canvases[-1, :], (config['B'], config['A']))
        input_image = np.reshape(1 - xnext, (config['B'], config['A']))
      arr[0, t].imshow(input_image, cmap='gray', vmin=0, vmax=1)
      arr[0, t].set_xticks([])
      arr[0, t].set_yticks([])
      arr[1, t].imshow(end_result, cmap='gray', vmin=0, vmax=1)
      arr[1, t].set_xticks([])
      arr[1, t].set_yticks([])
      
    plt.show()


if __name__ == '__main__':
    main(test_config)
