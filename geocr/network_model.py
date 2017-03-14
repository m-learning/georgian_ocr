"""
Created on Feb 18, 2017

Network model for OCR module

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import GRU
from keras.models import Model

from geocr.network_config import conv_num_filters, filter_size, pool_size, \
                                   rnn_size, time_dense_size, act
from geocr.network_config import init_img_gen, init_conv_to_rnn_dims, init_input_shape, \
                                   ctc_lambda_func
from geocr.training_flags import read_flags as flags


MODEL_YAML_FILE = 'ocr_model'

def init_model(img_w, output_size=28):
  """Initializes OCR network model
    Args:
      img_w - input image width
      img_get - image generator
    Returns:
      tuple of -
        input_data - network model input data
        network_model - network model
  """ 

  input_shape = init_input_shape(img_w)
  
  input_data = Input(name='the_input', shape=input_shape, dtype='float32')
  inner = Conv2D(conv_num_filters, kernel_size=(filter_size, filter_size), padding='same',
                 activation=act, kernel_initializer='he_normal', name='conv1')(input_data)
  inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
  inner = Conv2D(conv_num_filters, kernel_size=(filter_size, filter_size), padding='same',
                 activation=act, kernel_initializer='he_normal', name='conv2')(inner)
  inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

  conv_to_rnn_dims = init_conv_to_rnn_dims(img_w)
  inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

  # cuts down input size going into RNN:
  inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

  # Two layers of bidirecitonal GRUs
  # GRU seems to work as well, if not better than LSTM:
  gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
  gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
  gru1_merged = add([gru_1, gru_1b])
  gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
  gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
  gru2_merged = concatenate([gru_2, gru_2b])
  # transforms RNN output to character activations:
  inner = Dense(output_size, kernel_initializer='he_normal', name='dense2')(gru2_merged)
  network_model = Activation('softmax', name='softmax')(inner)
  
  return (input_data, network_model) 

def ocr_network(img_w):
  """Initializes OCR network model
    Args:
      img_w - image weight
    Returns:
      tuple of -
        input_data - network model input data
        y_pred - prediction model
        network_model - network model
  """
  
  img_gen = init_img_gen(img_w)
  output_size = img_gen.get_output_size()
  (input_data, y_pred) = init_model(img_w, output_size=output_size)
  network_model = Model(inputs=[input_data], outputs=[y_pred])
  
  return (img_gen, input_data, y_pred, network_model)

def _serialize_network(img_w, model):
  """Serializes network model to YAML file
    Args:
    img_w - input image width
    model - network model
  """

  model_yaml = model.to_yaml()
  with open(os.path.join(flags().model_dir, 'model%d.yaml' % img_w), "w") as yaml_file:
    yaml_file.write(model_yaml)

def _validate_and_serialize(img_w, model):
  """Serializes network model to YAML file
    Args:
    img_w - input image width
    model - network model
  """
      
  if flags().save_model:
    _serialize_network(img_w, model)

def init_training_model(img_w):
  """Initializes OCR network model
    Args:
      img_w - input image width
      img_get - image generator
    Returns:
      tuple of -
        input_data - model input data
        model - network model
        y_pred - prediction model
  """
  
  (img_gen, input_data, y_pred, network_model) = ocr_network(img_w)
  _validate_and_serialize(img_w, network_model)
  network_model.summary()

  labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
  input_length = Input(name='input_length', shape=[1], dtype='int64')
  label_length = Input(name='label_length', shape=[1], dtype='int64')
  # Keras doesn't currently support loss funcs with extra parameters
  # so CTC loss is implemented in a lambda layer
  loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
  # clipnorm seems to speeds up convergence
  model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])
  
  return ((y_pred, input_data), (model, img_gen))
