import numpy as np

from assignment2.cs231n.layers import *
from assignment2.cs231n.fast_layers import *
from assignment2.cs231n.layer_utils import *


def two_layer_convnet(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradient for a simple two-layer ConvNet. The architecture
  is conv-relu-pool-affine-softmax, where the conv layer uses stride-1 "same"
  convolutions to preserve the input size; the pool layer uses non-overlapping
  2x2 pooling regions. We use L2 regularization on both the convolutional layer
  weights and the affine layer weights.

  Inputs:
  - X: Input data, of shape (N, C, H, W)
  - model: Dictionary mapping parameter names to parameters. A two-layer Convnet
    expects the model to have the following parameters:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the affine layer
  - y: Vector of labels of shape (N,). y[i] gives the label for the point X[i].
  - reg: Regularization strength.

  Returns:
  If y is None, then returns:
  - scores: Matrix of scores, where scores[i, c] is the classification score for
    the ith input and class c.

  If y is not None, then returns a tuple of:
  - loss: Scalar value giving the loss.
  - grads: Dictionary with the same keys as model, mapping parameter names to
    their gradients.
  """
  
  # Unpack weights
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, C, H, W = X.shape

  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': int((conv_filter_height - 1) / 2)}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  scores, cache2 = affine_forward(a1, W2, b2)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da1, dW2, db2 = affine_backward(dscores, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
  
  return loss, grads


def init_two_layer_convnet(weight_scale=1e-3, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  """
  Initialize the weights for a two-layer ConvNet.

  Inputs:
  - weight_scale: Scale at which weights are initialized. Default 1e-3.
  - bias_scale: Scale at which biases are initialized. Default is 0.
  - input_shape: Tuple giving the input shape to the network; default is
    (3, 32, 32) for CIFAR-10.
  - num_classes: The number of classes for this network. Default is 10
    (for CIFAR-10)
  - num_filters: The number of filters to use in the convolutional layer.
  - filter_size: The width and height for convolutional filters. We assume that
    all convolutions are "same", so we pick padding to ensure that data has the
    same height and width after convolution. This means that the filter size
    must be odd.

  Returns:
  A dictionary mapping parameter names to numpy arrays containing:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the fully-connected layer.
  """
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['W2'] = weight_scale * np.random.randn(int(num_filters * H * W / 4), num_classes)
  model['b2'] = bias_scale * np.random.randn(num_classes)
  return model





def init_supercool_convnet(weight_scale=5e-2, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10):
  # 1 conv_relu_pool_forward
  # 2 conv_relu_pool_forward
  # 3 conv_relu_pool_forward
  # 4 affine_forward
  C, H, W = input_shape

  model = {}
  model['W1'] = weight_scale * np.random.randn(16, 3, 5, 5)
  model['W2'] = weight_scale * np.random.randn(20, 16, 5, 5)
  model['W3'] = weight_scale * np.random.randn(20, 20, 5, 5)
  model['W4'] = weight_scale * np.random.randn(20 * 4 * 4, num_classes)

  model['b1'] = bias_scale * np.random.randn(16)
  model['b2'] = bias_scale * np.random.randn(20)
  model['b3'] = bias_scale * np.random.randn(20)
  model['b4'] = bias_scale * np.random.randn(num_classes)
  return model


def supercool_convnet(X, model, y=None, reg=0.0):
  # 1 conv_relu_pool_forward
  # 2 conv_relu_pool_forward
  # 3 conv_relu_pool_forward
  # 4 affine_forward
  W1 = model['W1']
  W2 = model['W2']
  W3 = model['W3']
  W4 = model['W4']

  b1 = model['b1']
  b2 = model['b2']
  b3 = model['b3']
  b4 = model['b4']

  N, C, H, W = X.shape
  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  # Forward
  conv_param = {'stride': 1, 'pad': 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  out2, cache2 = conv_relu_pool_forward(out1, W2, b2, conv_param, pool_param)
  out3, cache3 = conv_relu_pool_forward(out2, W3, b3, conv_param, pool_param)
  scores, cache4 = affine_forward(out3, W4, b4)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  dout3, dW4, db4 = affine_backward(dscores, cache4)
  dout2, dW3, db3 = conv_relu_pool_backward(dout3, cache3)
  dout1, dW2, db2 = conv_relu_pool_backward(dout2, cache2)
  dX, dW1, db1 = conv_relu_pool_backward(dout1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'W2': dW2, 'W3': dW3, 'W4': dW4, 'b1': db1, 'b2': db2, 'b3': db3, 'b4': db4}

  return loss, grads


def init_supercoolbn_convnet(weight_scale=1e-5, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10):
  # 1 conv_relu_pool_forward
  # 2 conv_relu_pool_forward
  # 3 conv_relu_pool_forward
  # 4 affine_forward
  C, H, W = input_shape

  model = {}
  model['W1'] = weight_scale * np.random.randn(16, 3, 5, 5)
  model['W2'] = weight_scale * np.random.randn(20, 16, 5, 5)
  model['W3'] = weight_scale * np.random.randn(20, 20, 5, 5)
  model['W4'] = weight_scale * np.random.randn(20 * 4 * 4, num_classes)

  model['b1'] = bias_scale * np.random.randn(16)
  model['b2'] = bias_scale * np.random.randn(20)
  model['b3'] = bias_scale * np.random.randn(20)
  model['b4'] = bias_scale * np.random.randn(num_classes)

  model['gamma1'] = np.ones(16, dtype=model['W1'].dtype)
  model['beta1'] = np.zeros(16, dtype=model['W1'].dtype)
  model['gamma2'] = np.ones(20, dtype=model['W2'].dtype)
  model['beta2'] = np.zeros(20, dtype=model['W2'].dtype)
  model['gamma3'] = np.ones(20, dtype=model['W3'].dtype)
  model['beta3'] = np.zeros(20, dtype=model['W3'].dtype)

  model['bn_param1'] = {'mode': 'train', 'eps': 1e-3, 'momentum': 0.9,
                        'running_mean': np.zeros(16, dtype=model['W1'].dtype),
                        'running_var': np.ones(16, dtype=model['W1'].dtype)}
  model['bn_param2'] = {'mode': 'train', 'eps': 1e-2, 'momentum': 0.9,
                        'running_mean': np.zeros(20, dtype=model['W2'].dtype),
                        'running_var': np.ones(20, dtype=model['W2'].dtype)}
  model['bn_param3'] = {'mode': 'train', 'eps': 1e-2, 'momentum': 0.9,
                        'running_mean': np.zeros(20, dtype=model['W2'].dtype),
                        'running_var': np.ones(20, dtype=model['W2'].dtype)}


  return model


def supercoolbn_convnet(X, model, y=None, reg=0.0):
  # 1 conv_relu_pool_forward
  # 2 conv_relu_pool_forward
  # 3 conv_relu_pool_forward
  # 4 affine_forward
  W1 = model['W1']
  W2 = model['W2']
  W3 = model['W3']
  W4 = model['W4']

  b1 = model['b1']
  b2 = model['b2']
  b3 = model['b3']
  b4 = model['b4']

  gamma1, beta1 = model['gamma1'], model['beta1']
  gamma2, beta2 = model['gamma2'], model['beta2']
  gamma3, beta3 = model['gamma3'], model['beta3']
  bn_param1 = model['bn_param1']
  bn_param2 = model['bn_param2']
  bn_param3 = model['bn_param3']

  for params in {bn_param1, bn_param2, bn_param3}:
    params['mode'] = 'test' if y is None else 'train'

  N, C, H, W = X.shape
  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  # Forward
  conv_param = {'stride': 1, 'pad': 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  # out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  # out2, cache2 = conv_relu_pool_forward(out1, W2, b2, conv_param, pool_param)
  # out3, cache3 = conv_relu_pool_forward(out2, W3, b3, conv_param, pool_param)
  out1, cache1 = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, pool_param, bn_param1)
  out2, cache2 = conv_bn_relu_pool_forward(out1, W2, b2, gamma2, beta2, conv_param, pool_param, bn_param2)
  out3, cache3 = conv_bn_relu_pool_forward(out2, W3, b3, gamma3, beta3, conv_param, pool_param, bn_param3)
  scores, cache4 = affine_forward(out3, W4, b4)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  dout3, dW4, db4 = affine_backward(dscores, cache4)
  # dout2, dW3, db3 = conv_relu_pool_backward(dout3, cache3)
  # dout1, dW2, db2 = conv_relu_pool_backward(dout2, cache2)
  # dX, dW1, db1 = conv_relu_pool_backward(dout1, cache1)
  dout2, dW3, db3 = conv_bn_relu_pool_backward(dout3, cache3)
  dout1, dW2, db2 = conv_bn_relu_pool_backward(dout2, cache2)
  dX, dW1, db1 = conv_bn_relu_pool_backward(dout1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'W2': dW2, 'W3': dW3, 'W4': dW4, 'b1': db1, 'b2': db2, 'b3': db3, 'b4': db4}

  return loss, grads


def init_five_layer_convnet(input_shape=(3, 64, 64), num_classes=100,
                            filter_sizes=(5, 5, 5), num_filters=(32, 32, 64, 128),
                            weight_scale=1e-2, bias_scale=0, dtype=np.float32):
  """
  Initialize a five-layer convnet with the following architecture:

  [conv - relu - pool] x 3 - affine - relu - dropout - affine - softmax

  Each pooling region is 2x2 stride 2 and each convolution uses enough padding
  so that all convolutions are "same".

  Inputs:
  - Input shape: A tuple (C, H, W) giving the shape of each input that will be
    passed to the ConvNet. Default is (3, 64, 64) which corresponds to
    TinyImageNet.
  - num_classes: Number of classes over which classification will be performed.
    Default is 100 for TinyImageNet-100-A / TinyImageNet-100-B.
  - filter_sizes: Tuple of 3 integers giving the size of the filters for the
    three convolutional layers. Default is (5, 5, 5) which corresponds to 5x5
    filter at each layer.
  - num_filters: Tuple of 4 integers where the first 3 give the number of
    convolutional filters for the three convolutional layers, and the last
    gives the number of output neurons for the first affine layer.
    Default is (32, 32, 64, 128).
  - weight_scale: All weights will be randomly initialized from a Gaussian
    distribution whose standard deviation is weight_scale.
  - bias_scale: All biases will be randomly initialized from a Gaussian
    distribution whose standard deviation is bias_scale.
  - dtype: numpy datatype which will be used for this network. Float32 is
    recommended as it will make floating point operations faster.
  """
  C, H, W = input_shape
  F1, F2, F3, FC = num_filters
  # filter_size = 5
  model = {}
  model['W1'] = np.random.randn(F1, 3, filter_sizes[0], filter_sizes[0])
  model['b1'] = np.random.randn(F1)
  model['W2'] = np.random.randn(F2, F1, filter_sizes[1], filter_sizes[1])
  model['b2'] = np.random.randn(F2)
  model['W3'] = np.random.randn(F3, F2, filter_sizes[2], filter_sizes[2])
  model['b3'] = np.random.randn(F3)
  model['W4'] = np.random.randn(int(H * W * F3 / 64), FC)
  model['b4'] = np.random.randn(FC)
  model['W5'] = np.random.randn(FC, num_classes)
  model['b5'] = np.random.randn(num_classes)

  for i in [1, 2, 3, 4, 5]:
    model['W%d' % i] *= weight_scale
    model['b%d' % i] *= bias_scale

  for k in model:
    model[k] = model[k].astype(dtype, copy=False)

  return model


def five_layer_convnet(X, model, y=None, reg=0.0, dropout=0.5):
  """
  Compute the loss and gradient for a five layer convnet with the architecture

  [conv - relu - pool] x 3 - affine - relu - dropout - affine - softmax

  Each conv is stride 1 with padding chosen so the convolutions are "same";
  all padding is 2x2 stride 2.

  We use L2 regularization on all weight matrices and no regularization on
  biases.

  This function can output several different things:

  If y not given, then this function will output extracted features,
  classification scores, or classification probabilities depending on the
  values of the extract_features and return_probs flags.

  If y is given, then this function will output either (loss, gradients)
  or dX, depending on the value of the compute_dX flag.

  Inputs:
  - X: Input data of shape (N, C, H, W)
  - model: Dictionary mapping string names to model parameters. We expect the
    following parameters:
    W1, b1, W2, b2, W3, b3: Weights and biases for the conv layers
    W4, b4, W5, b5: Weights and biases for the affine layers
  - y: Integer vector of shape (N,) giving labels for the data points in X.
    If this is given then we will return one of (loss, gradient) or dX;
    If this is not given then we will return either class scores or class
    probabilities.
  - reg: Scalar value giving the strength of L2 regularization.
  - dropout: The probability of keeping a neuron in the dropout layer

  Outputs:
  This function can return several different things, depending on its inputs
  as described above.

  If y is None and extract_features is True, returns:
  - features: (N, H) array of features, where H is the number of neurons in the
    first affine layer.
  
  If y is None and return_probs is True, returns:
  - probs: (N, L) array of normalized class probabilities, where probs[i][j]
    is the probability that X[i] has label j.

  If y is None and return_probs is False, returns:
  - scores: (N, L) array of unnormalized class scores, where scores[i][j] is
    the score assigned to X[i] having label j.

  If y is not None and compute_dX is False, returns:
  - (loss, grads) where loss is a scalar value giving the loss and grads is a
    dictionary mapping parameter names to arrays giving the gradient of the
    loss with respect to each parameter.

  If y is not None and compute_dX is True, returns:
  - dX: Array of shape (N, C, H, W) giving the gradient of the loss with
    respect to the input data.
  """
  W1, b1 = model['W1'], model['b1']
  W2, b2 = model['W2'], model['b2']
  W3, b3 = model['W3'], model['b3']
  W4, b4 = model['W4'], model['b4']
  W5, b5 = model['W5'], model['b5']

  conv_param_1 = {'stride': 1, 'pad': int((W1.shape[2] - 1) / 2)}
  conv_param_2 = {'stride': 1, 'pad': int((W2.shape[2] - 1) / 2)}
  conv_param_3 = {'stride': 1, 'pad': int((W3.shape[2] - 1) / 2)}
  pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
  dropout_param = {'p': dropout}
  dropout_param['mode'] = 'test' if y is None else 'train'

  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param_1, pool_param)
  a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param_2, pool_param)
  a3, cache3 = conv_relu_pool_forward(a2, W3, b3, conv_param_3, pool_param)
  a4, cache4 = affine_relu_forward(a3, W4, b4)
  d4, cache5 = dropout_forward(a4, dropout_param)
  scores, cache6 = affine_forward(d4, W5, b5)

  if y is None:
      return scores

  data_loss, dscores = softmax_loss(scores, y)
  dd4, dW5, db5 = affine_backward(dscores, cache6)
  da4 = dropout_backward(dd4, cache5)
  da3, dW4, db4 = affine_relu_backward(da4, cache4)
  da2, dW3, db3 = conv_relu_pool_backward(da3, cache3)
  da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
  dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)


  grads = {
    'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2,
    'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4,
    'W5': dW5, 'b5': db5,
  }

  reg_loss = 0.0
  for p in ['W1', 'W2', 'W3', 'W4', 'W5']:
    W = model[p]
    reg_loss += 0.5 * reg * np.sum(W)
    grads[p] += reg * W
  loss = data_loss + reg_loss

  return loss, grads


def init_supercool2_convnet(weight_scale=5e-2, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=3):
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
  # /4 /8 if pooling once, /16, /32 if pooling twice
  # /16 /64 does better; less params at end
  model['W3'] = weight_scale * np.random.randn(int(num_filters * H * W / 16),
                                               int(num_filters * H * W / 64))  # after pooling, size is less than H*W, divided by 4
  model['W4'] = weight_scale * np.random.randn(int(num_filters * H * W / 64), num_classes)

  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['b2'] = bias_scale * np.random.randn(num_filters)
  model['b3'] = bias_scale * np.random.randn(int(num_filters * H * W / 64))
  model['b4'] = bias_scale * np.random.randn(num_classes)
  return model


def supercool2_convnet(X, model, y=None, reg=1e-4):
  # 1 conv_relu_forward
  # 2 conv_relu_pool_forward
  # 3 affine_forward
  # 4 affine_forward -- becomes SVM
  W1 = model['W1']
  W2 = model['W2']
  W3 = model['W3']
  W4 = model['W4']
  b1 = model['b1']
  b2 = model['b2']
  b3 = model['b3']
  b4 = model['b4']

  N, C, H, W = X.shape
  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': int((conv_filter_height - 1) / 2)}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  # Forward
  o1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  o2, cache2 = conv_relu_pool_forward(o1, W2, b2, conv_param, pool_param)
  o3, cache3 = affine_forward(o2, W3, b3)
  o4, cache5 = relu_forward(o3)
  scores, cache4 = affine_forward(o4, W4, b4)

  if y is None:
    return scores

  # Compute the backward pass
  # Make last layer linear SVM
  data_loss, dscores = svm_loss(scores, y)

  # Compute the gradients using a backward pass
  do4, dW4, db4 = affine_backward(dscores, cache4)
  do3 = relu_backward(do4, cache5)
  do2, dW3, db3 = affine_backward(do3, cache3)
  do1, dW2, db2 = conv_relu_pool_backward(do2, cache2)
  dX, dW1, db1 = conv_relu_pool_backward(do1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'W2': dW2, 'W3': dW3, 'W4': dW4, 'b1': db1, 'b2': db2, 'b3': db3, 'b4': db4}

  return loss, grads



# def init_supercool_convnet_big(weight_scale=5e-2, bias_scale=0, input_shape=(3, 32, 32),
#                                num_classes=10, num_filters=32, filter_size=3):
#   C, H, W = input_shape
#   assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size
#
#   model = {}
#   model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
#   model['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
#   model['W3'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
#   model['W4'] = weight_scale * np.random.randn(i)  # after pooling, size is less than H*W, divided by 4
#   model['W5'] = weight_scale * np.random.randn(num_filters * H * W / 8, num_classes)
#
#   model['b1'] = bias_scale * np.random.randn(num_filters)
#   model['b2'] = bias_scale * np.random.randn(num_filters)
#   model['b3'] = bias_scale * np.random.randn(num_filters)
#   model['b4'] = bias_scale * np.random.randn(num_filters * H * W / 8)
#   model['b5'] = bias_scale * np.random.randn(num_classes)
#   return model
#
#
# def supercool_convnet_big(X, model, y=None, reg=1e-4):
#   # 1 conv_relu_forward
#   # 2 conv_relu_pool_forward
#   # 3 affine_forward
#   # 4 affine_forward -- becomes SVM
#   W1 = model['W1']
#   W2 = model['W2']
#   W3 = model['W3']
#   W4 = model['W4']
#   W4 = model['W5']
#   b1 = model['b1']
#   b2 = model['b2']
#   b3 = model['b3']
#   b4 = model['b4']
#   b4 = model['b5']
#
#   N, C, H, W = X.shape
#   # We assume that the convolution is "same", so that the data has the same
#   # height and width after performing the convolution. We can then use the
#   # size of the filter to figure out the padding.
#   conv_filter_height, conv_filter_width = W1.shape[2:]
#   assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
#   assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
#   assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
#   conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
#   pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
#
#   # Forward
#   o1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
#   o2, cache2 = conv_relu_pool_forward(o1, W2, b2, conv_param, pool_param)
#   o2, cache2 = conv_relu_pool_forward(o1, W2, b2, conv_param, pool_param)
#   o3, cache3 = affine_forward(o2, W3, b3)
#   scores, cache4 = affine_forward(o3, W4, b4)
#
#   if y is None:
#     return scores
#
#   # Compute the backward pass
#   # Make last layer linear SVM
#   data_loss, dscores = softmax_loss(scores, y)
#
#   # Compute the gradients using a backward pass
#   do3, dW4, db4 = affine_backward(dscores, cache4)
#   do2, dW3, db3 = affine_backward(do3, cache3)
#   do1, dW2, db2 = conv_relu_pool_backward(do2, cache2)
#   dX, dW1, db1 = conv_relu_backward(do1, cache1)
#
#   # Add regularization
#   dW1 += reg * W1
#   dW2 += reg * W2
#   dW3 += reg * W3
#   dW4 += reg * W4
#   reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])
#
#   loss = data_loss + reg_loss
#   grads = {'W1': dW1, 'W2': dW2, 'W3': dW3, 'W4': dW4, 'b1': db1, 'b2': db2, 'b3': db3, 'b4': db4}
#
#   return loss, grads




# class CNN(object):
#
#   def __init__(self, weight_scale=1e-5, bias_scale=0, input_shape=(3, 32, 32),
#                            filter_size=(5, 3, 3), filter_nums=(16, 16, 16),
#                            hidden_dims = (),  num_classes=10):
#
#     self.params, self.conv_param, self.bn_param={}, {}, {}
#     C, H, W = input_shape
#     input_maps = C
#     for i in range(len(filter_nums)):
#       self.params['W'+str(i)] = weight_scale * np.random.randn(filter_nums[i], input_maps, filter_size[i], filter_size[i])
#       self.params['b'+str(i)] = bias_scale * np.random.randn(filter_nums[i])
#       self.params['gamma'+str(i)] = np.ones((1, filter_nums[i]))
#       self.params['beta'+str(i)] = np.zeros((1, filter_nums[i]))
#       self.bn_param[i] = {'mode': 'train', 'eps': 1e-2, 'momentum': 0.9,
#                             'running_mean': np.zeros(filter_nums[i]),
#                             'running_var': np.ones(filter_nums[i])}
#       self.conv_param[i] = {'stride': 1, 'pad': int((filter_size[i] - 1) / 2)}
#       input_maps = filter_nums[i]
#     if len(hidden_dims) == 0:
#       self.params['W'+str(len(filter_nums))] = weight_scale * np.random.randn(int(filter_nums[-1]*H*W/pow(4, len(filter_nums))), num_classes)
#       self.params['b'+str(len(filter_nums))] = bias_scale * np.random.randn(num_classes)
#     else:
#       self.params['W'+str(len(filter_nums))] = weight_scale * np.random.randn(int(filter_nums[-1]*H*W/pow(4, len(filter_nums))), hidden_dims[0])
#       self.params['b'+str(len(filter_nums))] = bias_scale * np.random.randn(hidden_dims[0])
#       for j in range(len(hidden_dims) - 1):
#         self.params['W'+str(len(filter_nums)+j+1)] = weight_scale * np.random.randn(hidden_dims[j], hidden_dims[j+1])
#         self.params['b'+str(len(filter_nums)+j+1)] = weight_scale * np.random.randn(hidden_dims[j], hidden_dims[j+1])
#       self.params['W'+str(len(filter_nums) + len(hidden_dims))] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
#       self.params['b'+str(len(filter_nums) + len(hidden_dims))] = bias_scale * np.random.randn(num_classes)
#     self.pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
#
#
#
#
#   def train(self, X, y = None, reg=0.0):
#
#     num_conv = len(self.conv_param)
#     num_hid = int((len(self.params) - 4 * num_conv - 2) / 2)
#
#     out, cache= {}, {}
#     out[0] = X
#     for i in range(num_conv):
#       out[i+1], cache[i] = conv_bn_relu_pool_forward(out[i], self.params['W'+str(i)], self.params['b'+str(i)],
#                                                      self.params['gamma'+str(i)], self.params['beta'+str(i)],
#                                                      self.conv_param[i], self.pool_param, self.bn_param[i])
#       if y is None:
#         self.bn_param[i]['mode'] = 'test'
#
#
#     dout, dW, db, dgamma, dbeta = {}, {}, {}, {}, {}
#     if num_hid > 0:
#       for j in range(num_hid):
#         out[num_conv+1+j], cache[num_conv+j] = affine_relu_forward(out[num_conv+j], self.params['W'+str(num_conv+j)], self.params['b'+str(num_conv+j)])
#     out[num_conv+num_hid+1], cache[num_hid+num_conv] = affine_forward(out[num_conv+num_hid], self.params['W'+str(num_hid+num_conv)],
#                                                                         self.params['b'+str(num_conv+num_hid)])
#
#     if y is None:
#       return out[num_conv + num_hid + 1]
#
#     data_loss, dscores = softmax_loss(out[num_hid+num_conv+1], y)
#
#     dout[num_conv+num_hid], dW[num_conv+num_hid], db[num_conv+num_hid] = affine_backward(dscores, cache[num_conv+num_hid])
#     if num_hid > 0:
#       for j in reversed(range(num_hid)):
#         dout[num_conv+j], dW[num_conv+j], db[num_conv+j] = affine_relu_backward(dout[num_conv+j+1], cache[num_conv+j])
#
#     for i in reversed(range(num_conv)):
#       dout[i], dW[i], db[i], dgamma[i], dbeta[i] = conv_bn_relu_pool_backward(dout[i+1], cache[i])
#
#     loss, reg_loss, grads = 0., 0., {}
#     for i in range(len(dW)):
#       dW[i] += reg * self.params['W'+str(i)]
#       reg_loss += 0.5 * reg * np.sum(self.params['W'+str(i)] ** 2)
#
#     loss = data_loss + reg_loss
#
#
#     for i in range(num_conv + num_hid + 1):
#       grads['W'+str(i)] = dW[i]
#       grads['b'+str(i)] = db[i]
#     for j in range(num_conv):
#       grads['gamma'+str(j)] = dgamma[j]
#       grads['beta'+str(j)] = dbeta[j]
#
#     return loss, grads


















  
