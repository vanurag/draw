import os
import tensorflow as tf
import numpy as np
from scipy import ndimage

from config import texture_config
from train import load_data, get_model_and_placeholders
import matplotlib.pyplot as plt

tf.flags.DEFINE_string("test_dir", "", "")
FLAGS = tf.flags.FLAGS


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
        ckpt_path = os.path.join(os.path.abspath(config['model_dir']), 'drawmodel.ckpt')
    else:
        ckpt_path = os.path.join(os.path.abspath(config['model_dir']), 'drawmodel-{}'.format(ckpt_id))
    print('Evaluating ' + ckpt_path)
    saver.restore(sess, ckpt_path)
    
    n_layers = 5
    f, arr = plt.subplots(2 + n_layers, 2 * n_test_textures)
    for t in range(n_test_textures):
      xnext = sess.run(next_texture)
      input_texture = np.reshape(xnext, (config['B'], config['A']))
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
        xref = np.reshape(L_ref, (1, config['img_size']))
        cref = np.zeros([config['batch_size'], config['img_size']])
        
        feed_dict = draw_model.get_feed_dict(xref, cref)
        fetches = {'canvases': draw_model.cs.stack(), 'read_bbs': draw_model.read_bb.stack(), 'write_bbs': draw_model.write_bb.stack()}
        test_out = sess.run(fetches, feed_dict)
        # results
        canvases = np.concatenate(test_out['canvases'])  # T x img_size
        read_bounding_boxes = np.concatenate(test_out['read_bbs'])  # T x 3
        write_bounding_boxes = np.concatenate(test_out['write_bbs'])  # T x 3
        
        draw_result = np.reshape(canvases[-1, :], (config['B'], config['A']))
        
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


if __name__ == '__main__':
    main(texture_config)
