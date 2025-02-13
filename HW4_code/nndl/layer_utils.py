from .layers import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def affine_relu_forward(x, w, b):
  """
  Convenience layer that performs an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  '''
  Forward pass for the affine-batchnorm-relu layers
  '''

  a, fc_cache = affine_forward(x, w, b)
  bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(bn)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def affine_batchnorm_relu_backward(dout, cache):
  '''
  Backward pass for the affine-batchnorm-relu layers
  '''
  fc_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  d_bn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
  dx, dw, db = affine_backward(d_bn, fc_cache)
  return dx, dw, db, dgamma, dbeta

def affine_batchnorm_relu_dropout_forward(x,w,b, gamma, beta, bn_param, dropout_param=None):
  '''
  Forward pass for the affine-batchnorm-relu-dropout layers
  '''
  a, fc_cache = affine_forward(x, w, b)
  bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(bn)
  if dropout_param is not None:
    bn_out = out.copy()
    out, dropout_cache = dropout_forward(bn_out, dropout_param)
    cache = (fc_cache, bn_cache, relu_cache, dropout_cache)
 
  else:
    cache = (fc_cache, bn_cache, relu_cache)

  return out, cache


def affine_batchnorm_relu_dropout_backward(dout, cache):
  '''
  Backward pass for the affine-batchnorm-relu layers
  '''
  fc_cache, bn_cache, relu_cache, dropout_cache = cache
  dh = dropout_backward(dout, dropout_cache)
  da = relu_backward(dh, relu_cache)
  d_bn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
  dx, dw, db = affine_backward(d_bn, fc_cache)
  
  return dx, dw, db, dgamma, dbeta

def affine_relu_dropout_forward(x, w, b, dropout_param=None):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)

    if dropout_param is not None:
      temp = out.copy()
      out, dropout_cache = dropout_forward(temp, dropout_param)
      cache = (fc_cache, relu_cache, dropout_cache)
      
    else: cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_dropout_backward(dout, cache):
  fc_cache, relu_cache, dropout_cache = cache
  
  dh = dropout_backward(dout, dropout_cache)
  da = relu_backward(dh, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  
  return dx, dw, db
