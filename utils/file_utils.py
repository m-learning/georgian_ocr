"""
Created on Jul 6, 2016

Utility module for training test and validation data files

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import types


try:
  from PIL import Image
except ImportError:
  print("Importing Image from PIL threw exception")
  import Image


# General parent directory for files
DATAS_DIR_NAME = 'datas'

# Training set archives directory suffix
TRAINIG_ZIP_FOLDER = 'training_arch'

# Files and directory constant parameters
PATH_FOR_PARAMETERS = 'trained_data'
PATH_FOR_TRAINING = 'training_data'
PATH_FOR_EVALUATION = 'eval_data'
PATH_FOR_TRAINING_PHOTOS = 'training_photos'

# Test files and directories
TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_NAME = 'test_image'

def count_files(dir_name):
  """Counts files in directory
    Args:
      dir_name - directory name
    Returns:
      file_count - amount of files
  """
  
  file_count = 0
  for _, _, files in os.walk(dir_name):
    file_count += len(files)
  
  return file_count

def file_exists(_file_path):
  """Validates if file exists for passed path
    Args:
      _file_path - file path
    Returns;
      boolean flag if file exists
  """
  
  return os.path.exists(_file_path)

def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.
    Args:
      dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

class files_and_path_utils(object):
  """Utility class for files and directories"""
  
  def __init__(self, parent_cnn_dir, path_to_training_photos=None):
    self.path_to_cnn_directory = os.path.join(DATAS_DIR_NAME, parent_cnn_dir)    
    if path_to_training_photos is None:
      self.path_to_training_photos = PATH_FOR_TRAINING_PHOTOS
    else:
      self.path_to_training_photos = path_to_training_photos
  
  def init_file_or_path(self, file_path):
    """Creates file if not exists
      Args:
        file_path - file path
      Returns:
        file_path - the same path
    """
    
    ensure_dir_exists(file_path)
    return file_path
  
  def join_path(self, path_inst, *other_path):
    """Joins passed file paths and function generating path
      Args:
        path_inst - file path or function
      Returns:
        generated - file path 
    """
    if isinstance(path_inst, types.StringType):
      init_path = path_inst
    else:
      init_path = path_inst()
    result = os.path.join(init_path, *other_path)
    
    return result
  
  def join_and_init_path(self, path_inst, *other_path):
    """Joins and creates file or directory paths
      Args:
        path_inst - image path or function 
                    returning path
        other_path - varargs for other paths
                     or functions
      Returns:
        result - joined path
    """
    
    result = self.join_path(path_inst, *other_path)
    self.init_file_or_path(result)
    
    return result
  
  def init_dir(self, dir_path, *other_path):
    """Creates appropriated directory 
       if such does not exists
      Args:
        dir_path - directory path
        *other_path - vavargs for other paths
                     or functions
      Returns:
        result_dir - joined directory path
    """
    
    result_dir = self.join_path(dir_path, *other_path)
    self.init_file_or_path(result_dir)
    
    return result_dir 
  
  def get_current(self):
    """Gets current directory of script
      Returns:
        current_dir - current directory
    """
      
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    dirs = os.path.split(current_dir)
    dirs = os.path.split(dirs[0])
    current_dir = dirs[0]
    
    return current_dir
  

class cnn_file_utils(files_and_path_utils):
  """Utility class for training and testing files and directories"""
  
  def __init__(self, parent_cnn_dir, path_to_training_photos=None, image_resizer=None):
    super(cnn_file_utils, self).__init__(parent_cnn_dir,
                                         path_to_training_photos=path_to_training_photos)
    self.image_resizer = image_resizer
    
  def read_image(self, pr):
    """Reads image with or without resizing
     Args:
      pr - image path
     Returns:
      im - resized image
    """
    
    if self.image_resizer is None:
      im = Image.open(pr)
    else:
      im = self.image_resizer.read_and_resize(pr)
      
    return im
  
  def write_image(self, im, n_im):
    """Writes image with or without resizing
      Args:
        im - image
        n_im - new image
    """
    if self.image_resizer is None:
      im.save(n_im)
    else:
      self.image_resizer.save_resized(im, n_im)
      
  def read_and_write(self, pr, n_im):
    """Reads and saves (resized or not) image from one path to other
      Args:
        pr - parent directory path
        n_im - new image
    """
    
    if self.image_resizer is None:
      im = Image.open(pr)
      im.save(n_im)
    else:
      self.image_resizer.read_resize_write(pr, n_im)
  
  def get_data_general_directory(self):
    """Gets or creates directories
      Returns:
        data directory
    """
    return self.join_and_init_path(self.get_current, self.path_to_cnn_directory)
  
  @property
  def data_root(self):
    """Root directory for datas
      Returns:
        datas root directory
    """
    return self.get_data_general_directory()
  
  def get_archives_directory(self):
    """Gets training set archives directory
      Args:
        training archives directory path
    """
    dest_directory = self.join_path(self.get_data_general_directory, TRAINIG_ZIP_FOLDER)
    ensure_dir_exists(dest_directory)
    
    return dest_directory
  
  def get_training_directory(self):
    """Gets training data directory
    Returns:
      training data directory path
    """
    return self.join_path(self.get_data_general_directory, PATH_FOR_TRAINING)

  def get_data_directory(self):
    """Gets directory for training set and parameters
      Returns:
        directory for training set and parameters
    """
    return self.join_path(self.get_training_directory, self.path_to_training_photos)
  
  def get_or_init_data_directory(self):
    """Creates directory for training set and parameters"""
    
    dir_path = self.get_data_directory()
    ensure_dir_exists(dir_path)
  
  @property
  def data_dir(self):
    """Creates directory for training set and parameters
      Returns:
        _data_dir - data directory path
    """
    
    _data_dir = self.get_training_directory()
    ensure_dir_exists(_data_dir)
    
    return _data_dir
  
  def data_file(self, _file_path):
    """Joins data directory path to passed file path
      Args:
        _file_path - data file path
      Returns:
        joined data directory and data file path
    """
    return self.join_path(self.data_dir, _file_path)
  
  def init_files_directory(self):
    """Gets or creates directory for trained parameters
      Returns:
        current_dir - directory for trained parameters
    """
      
    current_dir = self.join_path(self.get_data_general_directory, PATH_FOR_PARAMETERS)
    ensure_dir_exists(current_dir)
    
    return current_dir
  
  @property
  def model_dir(self):
    """Gets or creates directory for trained parameters
      Returns:
        current_dir - directory for trained parameters
    """
    
    return self.init_files_directory()
  
  def model_file(self, _file_path):
    """Joins models directory path to passed file path
      Args:
        _file_path - model file path
      Returns:
        joined models directory and model file path
    """
    return self.join_path(self.model_dir, _file_path)

  def get_or_init_test_dir(self):
    """Gets directory for test images
      Returns:
        current_dir - test images directory
    """
    
    current_dir = self.join_path(self.get_data_general_directory, TEST_IMAGES_DIR)
    ensure_dir_exists(current_dir)
    
    return current_dir
    
  def get_or_init_test_path(self):
    """Gets or initializes test image
      Returns:
        Test image path
    """
    return self.join_path(self.get_or_init_test_dir, TEST_IMAGE_NAME)
  
  def get_or_init_eval_path(self):
    """Gets / initializes evaluation directory
      Returns:
        Evaluation diractory path
    """
    return self.join_and_init_path(self.get_data_general_directory, PATH_FOR_EVALUATION)
  
  @property
  def eval_dir(self):
    """Gets validation data and files directory
      Returns:
        validation data directory
    """
    
    return self.get_or_init_eval_path()

def rename_files(partt, name, dir_path):
  """Renames files in directory
    Args:
      patt - file name pattern
      name - new name
      dir_path - path to directory
  """
  
  scan_path = os.path.join(dir_path , '*.jpg')
  for pr in glob.glob(scan_path):
    file_base_name = os.path.basename(pr)
    print(file_base_name)
    if file_base_name.startswith(partt):
      file_name = name + file_base_name
      full_file_name = os.path.join(dir_path, file_name)
      os.rename(pr, full_file_name)

def read_arguments_and_run():
  """Retrieves command line arguments for files processing"""
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--src_dir',
                          type=str,
                          help='Source directory.')
  arg_parser.add_argument('--pattern',
                          type=str,
                          help='File name pattern.') 
  arg_parser.add_argument('--name',
                          type=str,
                          default='rnmd__',
                          help='New file name prefix.')
  (argument_flags, _) = arg_parser.parse_known_args()
  if argument_flags.src_dir and argument_flags.pattern:
    rename_files(argument_flags.pattern,
                 argument_flags.name,
                 argument_flags.src_dir)
  
    
if __name__ == '__main__':
  """Converts images for training data set"""
  
  read_arguments_and_run()
