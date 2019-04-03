config = {}

# # MODEL PARAMETERS ##

# # MNIST
# config['A'] = 28  # image width
# config['B'] = 28  # image height
# config['img_size'] = config['B'] * config['A']  # the canvas size
# config['enc_size'] = 256  # number of hidden units / output size in LSTM
# config['dec_size'] = 256
# config['read_n'] = 3  # read glimpse grid width/height
# config['write_n'] = 1  # write glimpse grid width/height
# config['write_radius'] = 3
# config['read_attn'] = True
# config['read_size'] = 2 * config['read_n'] * config['read_n'] if config['read_attn'] else 2 * config['img_size']
# config['write_attn'] = True
# config['write_size'] = config['write_n'] * config['write_n'] if config['write_attn'] else config['img_size']
# config['z_size'] = 100  # QSampler output size
# config['T'] = 64  # MNIST generation sequence length
# config['batch_size'] = 100  # training minibatch size
# config['n_summary_per_batch'] = 10  # number of images summarized per batch
# config['train_iters'] = 10000
# config['save_checkpoints_every_epoch'] = int(0.05 * config['train_iters']) # save chpnt after atleast these many epochs
# config['learning_rate'] = 1e-3  # learning rate for optimizer
# config['eps'] = 1e-8  # epsilon for numerical stability
# config['draw_with_white'] = True;  # draw with white ink or black ink

# # ETH
config['A'] = 32  # image width
config['B'] = 32  # image height
config['img_size'] = config['B'] * config['A']  # the canvas size
config['draw_with_white'] = False;  # draw with white ink or black ink

config['enc_rnn_mode'] = 'BASIC'  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
config['enc_size'] = 400  # number of hidden units / output size in LSTM layer
config['n_enc_layers'] = 1  # number of layers in encoder LSTM
config['dec_rnn_mode'] = 'BASIC'  # The low level implementation of lstm cell. choose between "BASIC", "BLOCK" and "GRU"
config['dec_size'] = 400
config['n_dec_layers'] = 1  # number of layers in decoder LSTM
config['z_size'] = 200  # QSampler output size
config['T'] = 100  # MNIST generation sequence length

config['read_n'] = int(0.2 * max(config['A'], config['B']))  # read glimpse grid width/height
config['write_n'] = 1  # write glimpse grid width/height
config['write_radius'] = 0.125 * max(config['A'], config['B'])
config['read_attn'] = True
config['read_size'] = 2 * config['read_n'] * config['read_n'] if config['read_attn'] else 2 * config['img_size']
config['write_attn'] = True
config['write_size'] = config['write_n'] * config['write_n'] if config['write_attn'] else config['img_size']

config['batch_size'] = 100  # training minibatch size
config['n_summary_per_batch'] = 10
config['train_iters'] = 10000
config['save_checkpoints_every_epoch'] = int(0.05 * config['train_iters'])  # save chpnt after atleast these many epochs
config['learning_rate'] = 1e-3  # learning rate for optimizer
config['learning_rate_type'] = 'linear'  # ['fixed', 'exponential', 'linear']
config['learning_rate_decay_steps'] = 3000
config['learning_rate_decay_rate'] = 0.1
config['eps'] = 1e-8  # epsilon for numerical stability

# # DEBUG
# A, B = 4, 4  # image width,height
# img_size = B * A  # the canvas size
# enc_size = 5  # number of hidden units / output size in LSTM
# dec_size = 5
# read_n = 3  # read glimpse grid width/height
# write_n = 1  # write glimpse grid width/height
# write_radius = 2
# read_size = 2 * read_n * read_n if FLAGS.read_attn else 2 * img_size
# write_size = write_n * write_n if FLAGS.write_attn else img_size
# z_size = 5  # QSampler output size
# T = 5  # MNIST generation sequence length
# batch_size = 100  # training minibatch size
# config['n_summary_per_batch'] = 10
# train_iters = 500
# config['save_checkpoints_every_epoch'] = 300 # save chpnt after atleast these many epochs
# learning_rate = 1e-3  # learning rate for optimizer
# eps = 1e-8  # epsilon for numerical stability
# draw_with_white = False;  # draw with white ink or black ink
