import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

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
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  
  #unpacking the parameters
  pad_width, stride = conv_param['pad'], conv_param['stride']
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  #pad the input
  x_padded = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')

  #computing output dimensions
  out_h = 1 + (H + 2 * pad - HH) // stride
  out_w = 1 + (W + 2 * pad - WW) // stride
  out = np.zeros((N, F, out_h, out_w))      # initializing output array

  #implementing convolution
  
  for n in range(N):  # Iterate over each input sample
        for f in range(F):  # Iterate over each filter
            for i in range(out_h):
                for j in range(out_w):
                    # Extract region of interest
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    
                    region = x_padded[n, :, h_start:h_end, w_start:w_end]
                    
                    # Perform convolution (element-wise multiplication and summation)
                    out[n, f, i, j] = np.sum(region * w[f]) + b[f]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
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
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #

  x, w, b, conv_param = cache
  stride, pad = conv_param['stride'], conv_param['pad']

  #Extract shapes
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  _,_,out_h, out_w = dout.shape

  #Initialize gradients
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  #Pad the input and its gradient
  x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
  dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

  #Compute db - sum over batch, height and width
  db = np.sum(dout, axis=(0,2,3))

  # Compute dw (weight gradient) & dx (input gradient)
  for n in range(N):  # Loop over each input in the batch
    for f in range(F):  # Loop over each filter
      for i in range(out_h):  # Loop over output height
        h_start = i * stride
        h_end = h_start + HH

        for j in range(out_w):  # Loop over output width
          w_start = j * stride
          w_end = w_start + WW

          # Extract the local region of x_padded
          region = x_padded[n, :, h_start:h_end, w_start:w_end]

          # Compute dw (Gradient of loss w.r.t. filter weights)
          dw[f] += region * dout[n, f, i, j]

          # Compute dx_padded (Gradient of loss w.r.t. input)
          dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j]

    # Remove padding from dx
    dx = dx_padded[:, :, pad:H + pad, pad:W + pad]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

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
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #

  #unpacking the parameters
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']

  #computing output dimensions
  out_h = 1 + (H - pool_height)//stride
  out_w = 1 + (W- pool_width)//stride
  out = np.zeros((N, C, out_h, out_w))      # initializing output array

  #implementing convolution
  
  for n in range(N):  # Iterate over each input sample
    for c in range(C):
      for i in range(out_h):
          for j in range(out_w):
              # Extract region of interest
              h_start = i * stride
              h_end = h_start + pool_height
              w_start = j * stride
              w_end = w_start + pool_width
              
              out[n, c, i, j] = np.max(x[n, c, h_start:h_end, w_start:w_end])


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
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
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

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
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

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
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta