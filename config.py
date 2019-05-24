train_config = {}

# # MODEL PARAMETERS ##

# # MNIST
train_config['A'] = 28  # image width
train_config['B'] = 28  # image height
train_config['img_size'] = train_config['B'] * train_config['A']  # the canvas size
train_config['draw_with_white'] = True  # draw with white ink or black ink
train_config['draw_all_time'] = True  # Whether to stop drawing if the drawing looks close to target or keep drawing for T time steps
    
train_config['enc_rnn_mode'] = 'BASIC'  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
train_config['enc_size'] = 256  # number of hidden units / output size in LSTM layer
train_config['n_enc_layers'] = 1  # number of layers in encoder LSTM
train_config['dec_rnn_mode'] = 'BASIC'  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
train_config['dec_size'] = 256
train_config['n_dec_layers'] = 1  # number of layers in decoder LSTM
train_config['n_hidden_units'] = 0  # number of hidden units on the linear layer
train_config['z_size'] = 100  # QSampler output size
train_config['T'] = 64  # MNIST generation sequence length
    
train_config['read_n'] = 3  # read glimpse grid width/height
train_config['write_n'] = 1  # write glimpse grid width/height
train_config['write_radius'] = 3
train_config['read_attn'] = True
train_config['read_size'] = 2 * train_config['read_n'] * train_config['read_n'] if train_config['read_attn'] else 2 * train_config['img_size']
train_config['write_attn'] = True
train_config['write_size'] = train_config['write_n'] * train_config['write_n'] if train_config['write_attn'] else train_config['img_size']
    
train_config['batch_size'] = 100  # training minibatch size
train_config['n_summary_per_batch'] = 10
train_config['n_epochs'] = 50  # number of times the entire dataset is processed during training
train_config['save_checkpoints_every_epoch'] = 1  # save chpnt after atleast these many epochs
# HOTSTART: Initialize canvas as cropped input image
train_config['use_hot_start'] = False
train_config['crop_fraction_increase_rate'] = 0.2  # rate at which cropped part is grown each epoch
train_config['eps'] = 1e-8  # epsilon for numerical stability
train_config['annealing_schedules'] = {  # decay type: ['fixed', 'linear' or 'exponential']
  "learning_rate": {
    "init": 1e-4, "min": 1e-7,
    "decay_type": "fixed", "factor": 0.1, "iters": 5000,
    "staircase": True
  },
#   "write_decision_prior_log_odds": {
#     "init":2.71 ** -2, "min": 0.000000001,
#     "decay_type": "fixed", "factor": 0.1, "iters": 1000,
#     "staircase": False, "log": True
#   },
  "write_decision_prior_log_odds": {
    "init":-2.0, "min":-9.0,
    "decay_type": "fixed", "factor":-0.25, "iters": 1000,
    "staircase": False
  },
  "stop_writing_threshold": {
    "init": 25.0, "min": 15.0,
    "decay_type": "fixed", "factor": 0.2, "iters": 3000,
    "staircase": True
  },
}

# # ETH
# train_config['A'] = 32  # image width
# train_config['B'] = 32  # image height
# train_config['img_size'] = train_config['B'] * train_config['A']  # the canvas size
# train_config['draw_with_white'] = False  # draw with white ink or black ink
# train_config['draw_all_time'] = True  # Whether to stop drawing if the drawing looks close to target or keep drawing for T time steps
#     
# train_config['enc_rnn_mode'] = 'BASIC'  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
# train_config['enc_size'] = 400  # number of hidden units / output size in LSTM layer
# train_config['n_enc_layers'] = 1  # number of layers in encoder LSTM
# train_config['dec_rnn_mode'] = 'BASIC'  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
# train_config['dec_size'] = 400
# train_config['n_dec_layers'] = 1  # number of layers in decoder LSTM
# train_config['n_hidden_units'] = 64  # number of hidden units on the linear layer
# train_config['z_size'] = 200  # QSampler output size
# train_config['T'] = 100  # MNIST generation sequence length
#     
# train_config['read_n'] = int(0.2 * max(train_config['A'], train_config['B']))  # read glimpse grid width/height
# train_config['write_n'] = 1  # write glimpse grid width/height
# train_config['write_radius'] = 4  # 0.125 * max(train_config['A'], train_config['B'])
# train_config['read_attn'] = True
# train_config['read_size'] = 2 * train_config['read_n'] * train_config['read_n'] if train_config['read_attn'] else 2 * train_config['img_size']
# train_config['write_attn'] = True
# train_config['write_size'] = train_config['write_n'] * train_config['write_n'] if train_config['write_attn'] else train_config['img_size']
#     
# train_config['batch_size'] = 100  # training minibatch size
# train_config['n_summary_per_batch'] = 10
# train_config['n_epochs'] = 20  # number of times the entire dataset is processed during training
# train_config['save_checkpoints_every_epoch'] = 1  # save chpnt after atleast these many epochs
# train_config['use_hot_start'] = False  # Initialize canvas as cropped input image
# train_config['crop_fraction_increase_rate'] = 0.25  # rate at which cropped part is grown
# train_config['epochs_per_crop_fraction'] = 2  # TODO: number of epochs for each crop fraction 
# train_config['eps'] = 1e-8  # epsilon for numerical stability
# train_config['annealing_schedules'] = {  # decay type: ['fixed', 'linear' or 'exponential']
#   "learning_rate": {
#     "init": 1e-4, "min": 1e-7,
#     "decay_type": "linear", "factor": 0.1, "iters": 5000,
#     "staircase": True
#   },
#   "write_decision_prior_log_odds": {
#     "init": 10000.0, "min": 0.000000001,
#     "decay_type": "exponential", "factor": 0.1, "iters": 3000,
#     "staircase": False, "log": True
#   },
#   "stop_writing_threshold": {
#     "init": 45.0, "min": 15.0,
#     "decay_type": "linear", "factor": 0.2, "iters": 3000,
#     "staircase": True
#   },
# }

# # DEBUG
# train_config['A'] = 4  # image width
# train_config['B'] = 4  # image height
# train_config['img_size'] = train_config['B'] * train_config['A']  # the canvas size
# train_config['draw_with_white'] = False  # draw with white ink or black ink
# train_config['draw_all_time'] = True  # Whether to stop drawing if the drawing looks close to target or keep drawing for T time steps
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
# train_config['stop_writing_threshold'] = 10.0  # higher the threshold, longer the network is allowed to draw

test_config = train_config.copy()
test_config['model_dir'] = '/media/anurag/DATA-EXT/texture-synthesis/texture-datasets/ETH_Synthesizability/logs/DRAW_1556207845/'  # TODO path to the model that you want to evaluate
test_config['checkpoint_id'] = None  # if None, the last checkpoint will be used

# For layer-wise painting of a texture
texture_config = test_config.copy()
texture_config['batch_size'] = 1
texture_config['n_summary_per_batch'] = 1
