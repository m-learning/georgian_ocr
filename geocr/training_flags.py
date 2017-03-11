"""
Created on Feb 28, 2017

Flags for model training and evaluation

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime

import pylab

from geocr.cnn_files import training_file
import matplotlib as mpl


mpl.use('Agg')
pylab.ioff()

flag_args = None

def parse_eval_arguments():
  
  parser = argparse.ArgumentParser(description='Georgian OCR')
  parser.add_argument('--image',
                      metavar='image_path',
                      type=str,
                      help='Path to the image to recognize.')
  parser.add_argument('--weights',
                      metavar='weights_path',
                      type=str,
                      help='Path to the weights.')
  parser.add_argument('--width',
                      metavar='image_width',
                      type=int,
                      default=128,
                      help='image width: 128 or 512 (128 is default)')
  parser.add_argument('--model',
                      metavar='model',
                      type=str,
                      help='Path to model')
  (args, _) = parser.parse_args()
  
  return args

def parse_arguments():
  """Parses command line arguments
    Returns:
      args - parsed command line arguments
  """
  parser = argparse.ArgumentParser()
  _files = training_file()
  parser.add_argument('--img_width',
                      type=int,
                      default=128,
                      help='Input image width')
  parser.add_argument('--second_phase_width',
                      type=int,
                      default=256,
                      help='Input image width for second phase of training')
  parser.add_argument('--third_phase_width',
                      type=int,
                      default=512,
                      help='Input image width for second phase of training')
  parser.add_argument('--start_epoch',
                      type=int,
                      default=0,
                      help='Training start epoch')
  parser.add_argument('--stop_epoch',
                      type=int,
                      default=6,
                      help='Training stop epoch')
  parser.add_argument('--stop_second_phase',
                      type=int,
                      default=20,
                      help='Training stop epoch for second phase')
  parser.add_argument('--stop_third_phase',
                      type=int,
                      default=25,
                      help='Training stop epoch for third phase')
  parser.add_argument('--run_name',
                      type=str,
                      default=datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S'),
                      help='Training run name')
  parser.add_argument('--fonts',
                      action='append',
                      type=str,
                      help='Fonts for training')
  parser.add_argument('--major_font',
                      type=str,
                      help='Major font to use')
  parser.add_argument('--data_dir',
                      type=str,
                      default=_files.data_dir,
                      help='Data directory')
  parser.add_argument('--model_dir',
                      type=str,
                      default=_files.model_dir,
                      help='Model directory')
  parser.add_argument('--not_save_model',
                      dest='save_model',
                      action='store_false',
                      help='Saves network model to yaml file.')
  (args, _) = parser.parse_known_args()
  global flag_args
  flag_args = args
  
  return args
