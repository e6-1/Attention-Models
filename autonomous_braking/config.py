class Config(object):
  original_size = 32
  win_size = 32
  bandwidth = win_size**2
  batch_size = 32
  eval_batch_size = 50
  loc_std = 0.22
  img_width = 450
  img_height = 244
  num_channels = 3
  depth = 1
  sensor_size = win_size**2 * depth
  minRadius = 8
  hg_size = hl_size = 128
  g_size = 256
  cell_output_size = 256
  loc_dim = 2
  cell_size = 256
  cell_out_size = cell_size
  num_glimpses = 8
  num_classes = 2
  max_grad_norm = 5.

  step = 80
  lr_start = 1e-3
  lr_min = 1e-4

  # Monte Carlo sampling
  M = 10

  # loss weights
  lamda = 0.5

  # number of times to train
  num_train_iterations = 1
