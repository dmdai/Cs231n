import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  N = x.shape[0]
  out = x.reshape(N, np.prod(x.shape[1:])).dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  N = x.shape[0]
  db = np.sum(dout, axis=0)
  dw = x.reshape(N, np.prod(x.shape[1:])).T.dot(dout)
  dx = dout.dot(w.T).reshape(x.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout
  dx[x <= 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, H, W = x.shape
  F = w.shape[0]
  HH = w.shape[2]
  WW = w.shape[3]
  stride = conv_param['stride']
  pad = conv_param['pad']

  H1 = 1 + int((H + 2 * pad - HH) / stride)
  W1 = 1 + int((W + 2 * pad - WW) / stride)


  out = np.zeros((N, F, H1, W1))

  x_pad = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  for n in range(N):
    for f in range(F):
      for i in range(H1):
        for j in range(W1):
          x_window = x_pad[n, :, i * stride : i * stride + HH, j * stride : j * stride + WW]
          out[n, f, i, j] = np.sum(x_window * w[f]) + b[f] 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  N, F, H1, W1 = dout.shape
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  HH = w.shape[2]
  WW = w.shape[3]
  stride = conv_param['stride']
  pad = conv_param['pad']


  dx, dw, db = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)
  x_pad = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  dx_pad = np.pad(dx, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  db = np.sum(np.sum(np.sum(dout, axis=0),axis=1),axis=1)
  
  for n in range(N):
    for f in range(F):
      for i in range(H1):
        for j in range(W1):
          # Window we want to apply the respective f th filter over (C, HH, WW)
          x_window = x_pad[n, :, i * stride : i * stride + HH, j * stride : j * stride + WW]

          dw[f] += x_window * dout[n, f, i, j]
          # db[f] += dout[n, f, i, j]

          dx_pad[n, :, i * stride : i * stride + HH, j * stride : j * stride + WW] += w[f] * dout[n, f, i, j]
  
  dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']

  H1 = int((H - HH)) / stride + 1
  W1 = int((W - WW)) / stride + 1

  out = np.zeros([N, C, H1, W1])
  for n in range(N):
    for c in range(C):
      for i in range(H1):
        for j in range(W1):
          x_window = x[n, c, i * stride : i * stride + HH, j * stride : j * stride + WW]
          out[n, c, i, j] = np.max(x_window)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  N, C, H, W = x.shape
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']

  H1 = (H - HH) / stride + 1
  W1 = (W - WW) / stride + 1

  dx = np.zeros_like(x)
  for n in range(N):
    for c in range(C):
      for i in range(H1):
        for j in range(W1):
          x_window = x[n, c, i * stride : i * stride + HH, j * stride : j * stride + WW]
          x_max = np.max(x_window)
          #dx[n, c, i * stride : i * stride + HH, j * stride : j * stride + WW][x_window != x_max] = 0 
          dx[n, c, i * stride : i * stride + HH, j * stride : j * stride + WW]+=(x_window == x_max) * dout[n, c, i, j]


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We keep each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = np.random.rand(*x.shape)
    mask[mask >= p] = 0.
    mask[mask != 0] = 1.
    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def batchnorm_forward(X, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = X.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=X.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=X.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(X, axis=0, keepdims=True)                        # 1 by D
        sample_var = np.var(X, axis=0, keepdims=True)                          # 1 by D
        X_normalized = (X - sample_mean) / np.sqrt(sample_var + eps)            # N by D
        out = gamma * X_normalized + beta
        cache = (X_normalized, gamma, beta, sample_mean, sample_var, X, eps)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == 'test':
        X_normalized = (X - running_mean) / np.sqrt(running_var + eps)
        out = gamma * X_normalized + beta
    else:
        raise ValueError('Invalid forward batchnorm mode %s' %mode)

    # Store the updated running means back into bn_params
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dX, dgamma, dbeta = None, None, None
    X_normalized, gamma, beta, sample_mean, sample_var, X, eps = cache
    N, D = X.shape

    dX_normalized = dout * gamma                                         # N by D
    X_mu = X - sample_mean                                               # N by D
    sample_std_inv = 1.0 / np.sqrt(sample_var + eps)                     # 1 by D
    dsample_var = -0.5 * np.sum(dX_normalized * X_mu, axis=0, keepdims=True) * sample_std_inv**3       # 1 by D
    dsample_mean = -1.0 * np.sum(dX_normalized * sample_std_inv, axis=0, keepdims=True) -\
                   2.0 * dsample_var * np.mean(X_mu, axis=0, keepdims=True)

    dX = 1.0 * dsample_mean / N + 2.0 * dsample_var * X_mu / N + dX_normalized * sample_std_inv
    dgamma = np.sum(dout * X_normalized, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)

    return dX, dgamma, dbeta


# def batchnorm_forward(x, gamma, beta, bn_param):
#   """
#   Forward pass for batch normalization.
#
#   During training the sample mean and (uncorrected) sample variance are
#   computed from minibatch statistics and used to normalize the incoming data.
#   During training we also keep an exponentially decaying running mean of the mean
#   and variance of each feature, and these averages are used to normalize data
#   at test-time.
#
#   At each timestep we update the running averages for mean and variance using
#   an exponential decay based on the momentum parameter:
#
#   running_mean = momentum * running_mean + (1 - momentum) * sample_mean
#   running_var = momentum * running_var + (1 - momentum) * sample_var
#
#   Note that the batch normalization paper suggests a different test-time
#   behavior: they compute sample mean and variance for each feature using a
#   large number of training images rather than using a running average. For
#   this implementation we have chosen to use running averages instead since
#   they do not require an additional estimation step; the torch7 implementation
#   of batch normalization also uses running averages.
#
#   Input:
#   - x: Data of shape (N, D)
#   - gamma: Scale parameter of shape (D,)
#   - beta: Shift paremeter of shape (D,)
#   - bn_param: Dictionary with the following keys:
#     - mode: 'train' or 'test'; required
#     - eps: Constant for numeric stability
#     - momentum: Constant for running mean / variance.
#     - running_mean: Array of shape (D,) giving running mean of features
#     - running_var Array of shape (D,) giving running variance of features
#
#   Returns a tuple of:
#   - out: of shape (N, D)
#   - cache: A tuple of values needed in the backward pass
#   """
#   mode = bn_param['mode']
#   eps = bn_param.get('eps', 1e-5)
#   momentum = bn_param.get('momentum', 0.9)
#
#   N, D = x.shape
#   running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
#   running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
#
#   out, cache = None, None
#   if mode == 'train':
#     # Compute output
#     # mu = x.mean(axis=0)
#     mu = np.mean(x, axis=0)
#     xc = x - mu
#     var = np.mean(xc ** 2, axis=0)
#     std = np.sqrt(var + eps)
#     xn = xc / std
#     out = gamma * xn + beta
#
#     cache = (mode, x, gamma, xc, std, xn, out)
#
#     # Update running average of mean
#     running_mean *= momentum
#     running_mean += (1 - momentum) * mu
#
#     # Update running average of variance
#     running_var *= momentum
#     running_var += (1 - momentum) * var
#   elif mode == 'test':
#     # Using running mean and variance to normalize
#     std = np.sqrt(running_var + eps)
#     xn = (x - running_mean) / std
#     out = gamma * xn + beta
#     cache = (mode, x, xn, gamma, beta, std)
#   else:
#     raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
#
#   # Store the updated running means back into bn_param
#   bn_param['running_mean'] = running_mean
#   bn_param['running_var'] = running_var
#
#   return out, cache
#
#
# def batchnorm_backward(dout, cache):
#   """
#   Backward pass for batch normalization.
#
#   For this implementation, you should write out a computation graph for
#   batch normalization on paper and propagate gradients backward through
#   intermediate nodes.
#
#   Inputs:
#   - dout: Upstream derivatives, of shape (N, D)
#   - cache: Variable of intermediates from batchnorm_forward.
#
#   Returns a tuple of:
#   - dx: Gradient with respect to inputs x, of shape (N, D)
#   - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
#   - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
#   """
#   mode = cache[0]
#   if mode == 'train':
#     mode, x, gamma, xc, std, xn, out = cache
#
#     N = x.shape[0]
#     dbeta = dout.sum(axis=0)
#     dgamma = np.sum(xn * dout, axis=0)
#     dxn = gamma * dout
#     dxc = dxn / std
#     dstd = -np.sum((dxn * xc) / (std * std), axis=0)
#     dvar = 0.5 * dstd / std
#     dxc += (2.0 / N) * xc * dvar
#     dmu = np.sum(dxc, axis=0)
#     dx = dxc - dmu / N
#   elif mode == 'test':
#     mode, x, xn, gamma, beta, std = cache
#     dbeta = dout.sum(axis=0)
#     dgamma = np.sum(xn * dout, axis=0)
#     dxn = gamma * dout
#     dx = dxn / std
#   else:
#     raise ValueError(mode)
#
#   return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  N, C, H, W = x.shape
  x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
  out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
  out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  N, C, H, W = dout.shape
  dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
  dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
  dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  return dx, dgamma, dbeta


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx


def quadratic_loss(x, y):

  N = x.shape[0]
  loss = 0.5 * np.sum((x - y)**2) / N
  dx = (x - y) / N

  return loss, dx

