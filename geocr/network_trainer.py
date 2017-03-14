"""
Created on Feb 18, 2017

Module for OCR model training

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras import backend as K
from keras.optimizers import SGD

from geocr import network_config as config 
from geocr import network_model as network
from geocr import training_flags as flags
from geocr.data_generators import VizCallback
from geocr.network_config import OUTPUT_DIR
from geocr.network_config import words_per_epoch, val_words


def init_sgd_optimizer():
  """Initializes stochastic gradient descend optimizer"""
  return SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

def prepare_training(model, train_parameters):
  """Prepares model for training and loads weighs from file
    model - network model
    train_parameters - tuple of
      run_name - training run name
      start_epoch - training start epoch
  """
  
  (run_name, start_epoch, _) = train_parameters
  sgd = init_sgd_optimizer()
  model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
  if start_epoch > 0:
    weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
    model.load_weights(weight_file)
  
def train_model(network_parameters, train_parameters):
  """Trains network model
    Args:
      network_parameters - tuple of
        model - network model
        input_data - model input data
        y_pred - prediction model
        img_gen - image generator
      train_parameters - tuple of
        run_name - name of epochs
        start_epoch - epoch for start
        stop_epoch - epoch to stop
  """
  
  ((y_pred, input_data), (model, img_gen)) = network_parameters
  (run_name, start_epoch, stop_epoch) = train_parameters
  prepare_training(model, train_parameters)
  test_func = K.function([input_data], [y_pred])

  viz_cb = VizCallback(run_name, test_func, img_gen.next_val())

  model.fit_generator(generator=img_gen.next_train(), samples_per_epoch=(words_per_epoch - val_words),
                      nb_epoch=stop_epoch, validation_data=img_gen.next_val(), nb_val_samples=val_words,
                      callbacks=[viz_cb, img_gen], initial_epoch=start_epoch)

def train_network(run_name, start_epoch, stop_epoch, img_w):
  """Initializes and trains network model
    Args:
      run_name - name of epochs
      start_epoch - epoch for start
      stop_epoch - epoch to stop
      img_w - image width
  """
  
  network_parameters = network.init_training_model(img_w)
  train_parameters = (run_name, start_epoch, stop_epoch)
  train_model(network_parameters, train_parameters)
  
if __name__ == '__main__':
  """Train OCR model on custom images"""
  
  args = flags.parse_arguments()
  config.download_and_save()
  train_network(args.run_name, args.start_epoch, args.stop_epoch, args.img_width)
  # increase to wider images and start at epoch 20. The learned weights are reloaded
  train_network(args.run_name, args.stop_epoch, args.stop_second_phase, args.second_phase_width)
  train_network(args.run_name, args.stop_second_phase, args.stop_third_phase, args.third_phase_width)
