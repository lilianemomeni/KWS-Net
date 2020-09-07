# [filter size, stride, padding]
#Assume the two dimensions are the same
#Each kernel requires the following parameters:
# - k_i: kernel size
# - s_i: stride
# - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
#
#Each layer i requires the following parameters to be fully represented:
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i
# - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

import math
import numpy as np

def outFromIn(conv, layerIn, ceil_mode=True):
  n_in = layerIn[0]
  j_in = layerIn[1]
  r_in = layerIn[2]
  start_in = layerIn[3]
  k = conv[0]
  s = conv[1]
  p = conv[2]

  n_out = math.floor((n_in - k + 2*p)/s) + 1
  actualP = (n_out-1)*s - n_in + k
  pR = math.ceil(actualP/2)
  pL = math.floor(actualP/2)

  j_out = j_in * s
  r_out = r_in + (k - 1)*j_in
  start_out = start_in + ((k-1)/2 - pL)*j_in
  return n_out, j_out, r_out, start_out

def printLayer(layer, layer_name):
  print(layer_name + ":")
  print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))


strides_vid = np.array([2,6])
# --------- kern_size = 3 model with no paddings
convnet =   [[ 3,1,0]] * 8
convnet.append( [6,1,0] )
convnet = np.array(convnet)
convnet[strides_vid,0] = 5
layer_names = ['conv']*8 + ['fc']
# imsize = 270

def calc_receptive_field(layers, imsize, layer_names=None):
  if layer_names is not None: print ("-------Net summary------")
  currentLayer = [imsize, 1, 1, 0.5]
  if layer_names is not None: printLayer(currentLayer, "input image")
  # for i in range(len(convnet)):
  #   currentLayer = outFromIn(convnet[i], currentLayer)
  #   printLayer(currentLayer, layer_names[i])
  # print ("------------------------")


  for l_id, layer in enumerate(layers):
    conv = [ layer[key][-1] if type(layer[key]) in [list, tuple] else layer[key] for key in ['kernel_size', 'stride', 'padding'] ]
    currentLayer = outFromIn(conv, currentLayer)
    # print(conv)
    if layer_names is not None: printLayer(currentLayer, layer_names[l_id])
    if 'maxpool' in layer:
      conv = [ (layer['maxpool'][key][-1] if type(layer['maxpool'][key]) in [list, tuple] else layer['maxpool'][key])
               if (not key=='padding' or 'padding' in layer['maxpool'])
               else 0
               for key in ['kernel_size', 'stride', 'padding'] ]
      # import ipdb; ipdb.set_trace(context=20)
      currentLayer = outFromIn(conv, currentLayer, ceil_mode=False)
      # print(conv)
      if layer_names is not None: printLayer(currentLayer, layer_names[l_id] + '_maxpool')
  if layer_names is not None: print ("------------------------")
  return currentLayer


def main():
  layers = [
    { 'type': 'conv3d', 'n_channels': 32,  'kernel_size': (1,5,5), 'stride': (1,1,1), 'padding': (0,2,2)  ,   'maxpool': {'kernel_size' : (1,2,2), 'stride': (1,2,2)} },
    { 'type': 'conv3d', 'n_channels': 64, 'kernel_size': (1,5,5), 'stride': (1,2,2), 'padding': (0,2,2),   'maxpool': {'kernel_size' : (1,2,2), 'stride': (1,2,2)} },
    { 'type': 'conv3d', 'n_channels': 64, 'kernel_size': (1,2,2), 'stride': (1,2,2), 'padding': (0,0,0),   },
  ]
  # layer_names = ['conv{:d}'.format(lid) for lid in range(len(layers))]
  layer_names = None
  imsize = 361
  n_feats_out, jump, receptive_field, start_out = calc_receptive_field(layers, imsize, layer_names)
  print (imsize, n_feats_out)

if __name__ == '__main__':
  main()




