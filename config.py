train_config = {}

# # MODEL PARAMETERS ##

# # MNIST
# train_config['A'] = 28  # image width
# train_config['B'] = 28  # image height
# train_config['img_size'] = train_config['B'] * train_config['A']  # the canvas size
# train_config['draw_with_white'] = True  # draw with white ink or black ink
#   
# train_config['enc_rnn_mode'] = 'BASIC'  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
# train_config['enc_size'] = 256  # number of hidden units / output size in LSTM layer
# train_config['n_enc_layers'] = 1  # number of layers in encoder LSTM
# train_config['dec_rnn_mode'] = 'BASIC'  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
# train_config['dec_size'] = 256
# train_config['n_dec_layers'] = 1  # number of layers in decoder LSTM
# train_config['z_size'] = 100  # QSampler output size
# train_config['T'] = 64  # MNIST generation sequence length
#   
# train_config['read_n'] = 3  # read glimpse grid width/height
# train_config['write_n'] = 1  # write glimpse grid width/height
# train_config['write_radius'] = 3
# train_config['read_attn'] = True
# train_config['read_size'] = 2 * train_config['read_n'] * train_config['read_n'] if train_config['read_attn'] else 2 * train_config['img_size']
# train_config['write_attn'] = True
# train_config['write_size'] = train_config['write_n'] * train_config['write_n'] if train_config['write_attn'] else train_config['img_size']
#   
# train_config['batch_size'] = 100  # training minibatch size
# train_config['n_summary_per_batch'] = 10
# train_config['n_epochs'] = 20  # number of times the entire dataset is processed during training
# train_config['save_checkpoints_every_epoch'] = 1  # save chpnt after atleast these many epochs
# train_config['learning_rate'] = 1e-4  # learning rate for optimizer
# train_config['learning_rate_type'] = 'fixed'  # ['fixed', 'exponential', 'linear']
# train_config['learning_rate_decay_steps'] = 3000
# train_config['learning_rate_decay_rate'] = 0.1
# HOTSTART: Initialize canvas as cropped input image
# train_config['use_hot_start'] = False
# train_config['crop_fraction_increase_rate'] = 0.2  # rate at which cropped part is grown each epoch
# train_config['eps'] = 1e-8  # epsilon for numerical stability

# # ETH
train_config['A'] = 32  # image width
train_config['B'] = 32  # image height
train_config['img_size'] = train_config['B'] * train_config['A']  # the canvas size
train_config['draw_with_white'] = False  # draw with white ink or black ink
   
train_config['enc_rnn_mode'] = 'BASIC'  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
train_config['enc_size'] = 400  # number of hidden units / output size in LSTM layer
train_config['n_enc_layers'] = 1  # number of layers in encoder LSTM
train_config['dec_rnn_mode'] = 'BASIC'  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
train_config['dec_size'] = 400
train_config['n_dec_layers'] = 1  # number of layers in decoder LSTM
train_config['z_size'] = 200  # QSampler output size
train_config['T'] = 100  # MNIST generation sequence length
   
train_config['read_n'] = int(0.2 * max(train_config['A'], train_config['B']))  # read glimpse grid width/height
train_config['write_n'] = 1  # write glimpse grid width/height
train_config['write_radius'] = 4  # 0.125 * max(train_config['A'], train_config['B'])
train_config['read_attn'] = True
train_config['read_size'] = 2 * train_config['read_n'] * train_config['read_n'] if train_config['read_attn'] else 2 * train_config['img_size']
train_config['write_attn'] = True
train_config['write_size'] = train_config['write_n'] * train_config['write_n'] if train_config['write_attn'] else train_config['img_size']
   
train_config['batch_size'] = 100  # training minibatch size
train_config['n_summary_per_batch'] = 10
train_config['n_epochs'] = 10  # number of times the entire dataset is processed during training
train_config['save_checkpoints_every_epoch'] = 1  # save chpnt after atleast these many epochs
train_config['learning_rate'] = 1e-3  # learning rate for optimizer
train_config['learning_rate_type'] = 'fixed'  # ['fixed', 'exponential', 'linear']
train_config['learning_rate_decay_steps'] = 5000
train_config['learning_rate_decay_rate'] = 0.01
# HOTSTART: Initialize canvas as cropped input image
train_config['use_hot_start'] = True
train_config['crop_fraction_increase_rate'] = 0.2  # rate at which cropped part is grown each epoch
train_config['eps'] = 1e-8  # epsilon for numerical stability

# # DEBUG
# train_config['A'] = 4  # image width
# train_config['B'] = 4  # image height
# train_config['img_size'] = train_config['B'] * train_config['A']  # the canvas size
# train_config['draw_with_white'] = False  # draw with white ink or black ink
#  
# train_config['enc_rnn_mode'] = 'BASIC'  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
# train_config['enc_size'] = 5  # number of hidden units / output size in LSTM layer
# train_config['n_enc_layers'] = 1  # number of layers in encoder LSTM
# train_config['dec_rnn_mode'] = 'BASIC'  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
# train_config['dec_size'] = 5
# train_config['n_dec_layers'] = 1  # number of layers in decoder LSTM
# train_config['z_size'] = 5  # QSampler output size
# train_config['T'] = 5  # MNIST generation sequence length
#  
# train_config['read_n'] = 3  # read glimpse grid width/height
# train_config['write_n'] = 1  # write glimpse grid width/height
# train_config['write_radius'] = 2  # 0.125 * max(train_config['A'], train_config['B'])
# train_config['read_attn'] = True
# train_config['read_size'] = 2 * train_config['read_n'] * train_config['read_n'] if train_config['read_attn'] else 2 * train_config['img_size']
# train_config['write_attn'] = True
# train_config['write_size'] = train_config['write_n'] * train_config['write_n'] if train_config['write_attn'] else train_config['img_size']
#  
# train_config['batch_size'] = 100  # training minibatch size
# train_config['n_summary_per_batch'] = 2
# train_config['n_epochs'] = 2  # number of times the entire dataset is processed during training
# train_config['save_checkpoints_every_epoch'] = 300  # save chpnt after atleast these many epochs
# train_config['learning_rate'] = 1e-4  # learning rate for optimizer
# train_config['learning_rate_type'] = 'fixed'  # ['fixed', 'exponential', 'linear']
# train_config['learning_rate_decay_steps'] = 5000
# train_config['learning_rate_decay_rate'] = 0.1
# HOTSTART: Initialize canvas as cropped input image
# train_config['use_hot_start'] = True
# train_config['crop_fraction_increase_rate'] = 0.5  # rate at which cropped part is grown each epoch
# train_config['eps'] = 1e-8  # epsilon for numerical stability

test_config = train_config.copy()
test_config['batch_size'] = 1
test_config['n_summary_per_batch'] = 1
test_config['model_dir'] = '/media/anurag/DATA-EXT/texture-synthesis/texture-datasets/ETH_Synthesizability/logs/DRAW_1554726346/'  # TODO path to the model that you want to evaluate
test_config['checkpoint_id'] = '2'  # if None, the last checkpoint will be used
