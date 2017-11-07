
import numpy as np
import matplotlib.pyplot as plt
from assignment2.cs231n.classifiers.convnet import *
import numpy as np


def int_mnist_convnet(input_size=(1, 28, 28), weight_scale=1e-3):

  model = {}
  model['W1'] = np.random.randn(20, 1, 3, 3) * weight_scale
  model['b1'] = np.random.randn(20)
  model['W2'] = np.random.randn(20, 20, 3, 3) * weight_scale
  model['b2'] = np.random.randn((20))

  model['W3'] = np.random.randn(3920, 512) * weight_scale
  model['b3'] = np.random.randn(512)
  model['W4'] = np.random.randn(512, 10) * weight_scale
  model['b4'] = np.random.randn(10)

  return model


def mnist_convnet(X, model, y=None, reg=0.0):

  W1, b1 = model['W1'], model['b1']
  W2, b2 = model['W2'], model['b2']
  W3, b3 = model['W3'], model['b3']
  W4, b4 = model['W4'], model['b4']

  conv_param = {'stride': 1, 'pad': 1}
  pool_param = {'stride': 2, 'pool_width': 2, 'pool_height': 2}
  dropout_param = {'mode': 'train', 'p': 0.5}
  dropout_param['mode'] = 'test' if y is None else 'train'

  out1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
  out2, cache2 = conv_relu_pool_forward(out1, W2, b2, conv_param, pool_param)
  out3, cache3 = affine_relu_forward(out2, W3, b3)
  out3, cache_bp = dropout_forward(out3, dropout_param)
  out4, cache4 = affine_forward(out3, W4, b4)

  if y is None:
    return out4

  data_loss, dout4 = softmax_loss(out4, y)

  grads = {}
  dout3, grads['W4'], grads['b4'] = affine_backward(dout4, cache4)
  dout3 = dropout_backward(dout3, cache_bp)
  dout2, grads['W3'], grads['b3'] = affine_relu_backward(dout3, cache3)
  dout1, grads['W2'], grads['b2'] = conv_relu_pool_backward(dout2, cache2)
  dX, grads['W1'], grads['b1'] = conv_relu_backward(dout1, cache1)

  reg_loss = 0.
  for p in model:
    grads[p] += reg * model[p]
    reg_loss += 0.5 * reg * np.sum(model[p]**2)

  loss = reg_loss + data_loss

  return loss, grads




def init_twolayer_net(input_size=28*28, hidden_dim=100, num_classes=10, weight_scale=1e-4):

  model = {}
  model['W1'] = np.random.randn(input_size, hidden_dim) * weight_scale
  model['b1'] = np.random.randn(hidden_dim)
  model['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
  model['b2'] = np.random.randn(num_classes)

  return model



def twolayer_net(X, model, y=None, reg=0.):

  W1, b1 = model['W1'], model['b1']
  W2, b2 = model['W2'], model['b2']

  dropout_param = {'mode': 'train', 'p': 0.5}
  dropout_param['mode'] = 'test' if y is None else 'train'

  out1, cache1 = affine_relu_forward(X, W1, b1)
  out1, cache_bp = dropout_forward(out1, dropout_param)
  out2, cache2 = affine_forward(out1, W2, b2)

  if y is None:
    return out2

  grads = {}

  data_loss, dout2 = softmax_loss(out2, y)

  dout1, grads['W2'], grads['b2'] = affine_backward(dout2, cache2)
  dout1 = dropout_backward(dout1, cache_bp)
  dX, grads['W1'], grads['b1'] = affine_relu_backward(dout1, cache1)

  reg_loss = 0.
  for p in model:
    grads[p] += reg * model[p]
    reg_loss += 0.5 * reg * np.sum(model[p]**2)

  loss = reg_loss + data_loss

  return loss, grads


def ae_twolayer_net(X, model, y=None, reg=0.):

  W1, b1 = model['W1'], model['b1']
  W2, b2 = model['W2'], model['b2']

  out1, cache1 = affine_relu_forward(X, W1, b1)
  out2, cache2 = affine_relu_forward(out1, W2, b2)

  if y is None:
    return out2

  grads = {}

  data_loss, dout2 = quadratic_loss(out2, X)

  dout1, grads['W2'], grads['b2'] = affine_relu_backward(dout2, cache2)
  dX, grads['W1'], grads['b1'] = affine_relu_backward(dout1, cache1)

  reg_loss = 0.
  for p in model:
    grads[p] += reg * model[p]
    reg_loss += 0.5 * reg * np.sum(model[p]**2)

  loss = reg_loss + data_loss

  return loss, grads
