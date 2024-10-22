"""
Created on Jun 21, 2016

Files for training data

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.files.file_utils import cnn_file_utils


class training_file(cnn_file_utils):
  """Files and directories for parameters (trained), 
    training, validation and test"""
  
  def __init__(self):
    super(training_file, self).__init__('ocr')
