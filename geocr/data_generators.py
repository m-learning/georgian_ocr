#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mar 10, 2017

Data generators for OCR training

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import re

import editdistance
from keras import backend as K
import keras.callbacks
from keras.preprocessing import image
import matplotlib as mpl
mpl.use('Agg')
import pylab
from scipy import ndimage

import cairocffi as cairo
from geocr import font_storadge as _fonts
from geocr.character_translator import translate_to_geo
from geocr.cnn_files import training_file
import numpy as np


_files = training_file()
OUTPUT_DIR = _files.model_dir
IMG_DIR = _files.join_and_init_path(_files.data_root, 'images')

# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1
def speckle(img):
    
  severity = np.random.uniform(0, 0.6)
  blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
  img_speck = (img + blur)
  img_speck[img_speck > 1] = 1
  img_speck[img_speck <= 0] = 0
  
  return img_speck

def paint_text(text, w, h, rotate=False, ud=False, multi_fonts=False):
  """Paints the string in a random location the bounding box
    also uses a random font, a slight random rotation,
    and a random amount of speckle noise
   """
    
  surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
  with cairo.Context(surface) as context:
    context.set_source_rgb(1, 1, 1)  # White
    context.paint()
    # this font list works in Centos 7
    if multi_fonts:
      fonts = _fonts.georgian_fonts
      context.select_font_face(np.random.choice(fonts), cairo.FONT_SLANT_NORMAL,
                               np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
    else:
      # context.select_font_face('Courier', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
      context.select_font_face('Sylfaen', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    context.set_font_size(25)
    box = context.text_extents(translate_to_geo(text))
    border_w_h = (4, 4)
    if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
      raise IOError('Could not fit string into image. Max char count is too large for given image width.')

    # teach the RNN translational invariance by
    # fitting text box randomly on canvas, with some room to rotate
    max_shift_x = w - box[2] - border_w_h[0]
    max_shift_y = h - box[3] - border_w_h[1]
    top_left_x = np.random.randint(0, int(max_shift_x))
    if ud:
      top_left_y = np.random.randint(0, int(max_shift_y))
    else:
      top_left_y = h // 2
    context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
    context.set_source_rgb(0, 0, 0)
    context.show_text(translate_to_geo(text))

  buf = surface.get_data()
  a = np.frombuffer(buf, np.uint8)
  a.shape = (h, w, 4)
  a = a[:, :, 0]  # grab single channel
  a = a.astype(np.float32) / 255
  a = np.expand_dims(a, 0)
  if rotate:
    a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
  a = speckle(a)

  return a

def shuffle_mats_or_lists(matrix_list, stop_ind=None):
  """Shuffles mats or lists"""
    
  ret = []
  assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
  len_val = len(matrix_list[0])
  if stop_ind is None:
    stop_ind = len_val
  assert stop_ind <= len_val

  a = range(stop_ind)
  np.random.shuffle(a)
  a += range(stop_ind, len_val)
  for mat in matrix_list:
    if isinstance(mat, np.ndarray):
      ret.append(mat[a])
    elif isinstance(mat, list):
      ret.append([mat[i] for i in a])
    else:
      raise TypeError('shuffle_mats_or_lists only supports '
                        'numpy.array and list objects')
  return ret


def text_to_labels(text, num_classes):
    
  ret = []
  
  for char in text:
    if char >= 'a' and char <= 'z':
      ret.append(ord(char) - ord('a'))
    elif char == ' ':
      ret.append(26)
  
  return ret

def is_valid_str(in_str):
  """Only a-z and space..probably not to difficult
    to expand to uppercase and symbols
  """
  search = re.compile(r'^[ა-ჰ ]*$').search
  return not bool(search(in_str))

class TextImageGenerator(keras.callbacks.Callback):
  """Uses generator functions to supply train/test with
    data. Image renderings are text are created on the fly
    each time with random perturbations
  """

  def __init__(self, monogram_file, bigram_file, minibatch_size,
               img_w, img_h, downsample_factor, val_split,
               absolute_max_string_len=16):

    self.minibatch_size = minibatch_size
    self.img_w = img_w
    self.img_h = img_h
    self.monogram_file = monogram_file
    self.bigram_file = bigram_file
    self.downsample_factor = downsample_factor
    self.val_split = val_split
    self.blank_label = self.get_output_size() - 1
    self.absolute_max_string_len = absolute_max_string_len

  def get_output_size(self):
    return 34

  def build_word_list(self, num_words, max_string_len=None, mono_fraction=0.5):
    """Num_words can be independent of the epoch size due to the use of generator"""
    
    assert max_string_len <= self.absolute_max_string_len
    assert num_words % self.minibatch_size == 0
    assert (self.val_split * num_words) % self.minibatch_size == 0
    self.num_words = num_words
    self.string_list = [''] * self.num_words
    tmp_string_list = []
    self.max_string_len = max_string_len
    self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
    self.X_text = []
    self.Y_len = [0] * self.num_words

    # monogram file is sorted by frequency in english speech
    with open(self.monogram_file, 'rt') as f:
      for line in f:
          if len(tmp_string_list) == int(self.num_words * mono_fraction):
              break
          word = line.rstrip()
          if max_string_len == -1 or max_string_len is None or len(word) <= max_string_len:
              tmp_string_list.append(word)

    # bigram file contains common word pairings in english speech
    with open(self.bigram_file, 'rt') as f:
      lines = f.readlines()
      for line in lines:
          if len(tmp_string_list) == self.num_words:
              break
          columns = line.lower().split()
          word = columns[0] + ' ' + columns[1]
          if is_valid_str(word) and \
                  (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
              tmp_string_list.append(word)
    if len(tmp_string_list) != self.num_words:
      raise IOError('Could not pull enough words from supplied monogram and bigram files. ')
    # interlace to mix up the easy and hard words
    self.string_list[::2] = tmp_string_list[:self.num_words // 2]
    self.string_list[1::2] = tmp_string_list[self.num_words // 2:]

    for i, word in enumerate(self.string_list):
      self.Y_len[i] = len(word)
      self.Y_data[i, 0:len(word)] = text_to_labels(word, self.get_output_size())
      self.X_text.append(word)
    self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

    self.cur_val_index = self.val_split
    self.cur_train_index = 0

  def get_batch(self, index, size, train):
    """Each time an image is requested from train/val/test, a new random
      painting of the text is performed
    """
    # width and height are backwards from typical Keras convention
    # because width is the time dimension when it gets fed into the RNN
    if K.image_dim_ordering() == 'th':
      X_data = np.ones([size, 1, self.img_w, self.img_h])
    else:
      X_data = np.ones([size, self.img_w, self.img_h, 1])

    labels = np.ones([size, self.absolute_max_string_len])
    input_length = np.zeros([size, 1])
    label_length = np.zeros([size, 1])
    source_str = []
    for i in range(0, size):
      # Mix in some blank inputs.  This seems to be important for
    # achieving translational invariance
      if train and i > size - 4:
        if K.image_dim_ordering() == 'th':
            X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
        else:
            X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
        labels[i, 0] = self.blank_label
        input_length[i] = self.img_w // self.downsample_factor - 2
        label_length[i] = 1
        source_str.append('')
      else:
        if K.image_dim_ordering() == 'th':
            X_data[i, 0, 0:self.img_w, :] = self.paint_func(self.X_text[index + i])[0, :, :].T
        else:
            X_data[i, 0:self.img_w, :, 0] = self.paint_func(self.X_text[index + i])[0, :, :].T
        labels[i, :] = self.Y_data[index + i]
        input_length[i] = self.img_w // self.downsample_factor - 2
        label_length[i] = self.Y_len[index + i]
      source_str.append(self.X_text[index + i])
    inputs = {'the_input': X_data,
              'the_labels': labels,
              'input_length': input_length,
              'label_length': label_length,
              'source_str': source_str  # used for visualization only
              }
    outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
    
    return (inputs, outputs)

  def next_train(self):
    while 1:
      ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
      self.cur_train_index += self.minibatch_size
      if self.cur_train_index >= self.val_split:
        self.cur_train_index = self.cur_train_index % 32
        (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
          [self.X_text, self.Y_data, self.Y_len], self.val_split)
      yield ret

  def next_val(self):
    while 1:
      ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
      self.cur_val_index += self.minibatch_size
      if self.cur_val_index >= self.num_words:
        self.cur_val_index = self.val_split + self.cur_val_index % 32
      yield ret

  def on_train_begin(self, logs={}):
    self.build_word_list(16000, 4, 1)
    self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                              rotate=False, ud=False, multi_fonts=False)

  def on_epoch_begin(self, epoch, logs={}):
      # rebind the paint function to implement curriculum learning
    if epoch >= 3 and epoch < 6:
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                  rotate=False, ud=True, multi_fonts=False)
    elif epoch >= 6 and epoch < 9:
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                  rotate=False, ud=True, multi_fonts=True)
    elif epoch >= 9:
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                  rotate=True, ud=True, multi_fonts=True)
    if epoch >= 21 and self.max_string_len < 12:
        self.build_word_list(32000, 12, 0.5)

def ctc_lambda_func(args):
  """The actual loss calc occurs here despite it not being
    an internal Keras loss function
  """
  y_pred, labels, input_length, label_length = args
  # the 2 is critical here since the first couple outputs of the RNN
  # tend to be garbage:
  y_pred = y_pred[:, 2:, :]
  return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def decode_batch(test_func, word_batch):
  """For a real OCR application, this should be beam search with a dictionary
    and language model.  For this example, best path is sufficient.
  """
  
  out = test_func([word_batch])[0]
  ret = []
  for j in range(out.shape[0]):
      out_best = list(np.argmax(out[j, 2:], 1))
      out_best = [k for k, _ in itertools.groupby(out_best)]
      # 26 is space, 27 is CTC blank char
      outstr = ''
      for c in out_best:
          if c >= 0 and c < 26:
              outstr += chr(c + ord('a'))
          elif c == 26:
              outstr += ' '
      ret.append(outstr)
  
  return ret

class VizCallback(keras.callbacks.Callback):

  def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
    
    self.test_func = test_func
    self.output_dir = os.path.join(OUTPUT_DIR, run_name)
    self.img_dir = os.path.join(IMG_DIR, run_name)
    _files.join_and_init_path(self.img_dir)
    self.text_img_gen = text_img_gen
    self.num_display_words = num_display_words
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

  def show_edit_distance(self, num):
    num_left = num
    mean_norm_ed = 0.0
    mean_ed = 0.0
    while num_left > 0:
      word_batch = next(self.text_img_gen)[0]
      num_proc = min(word_batch['the_input'].shape[0], num_left)
      decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
      for j in range(0, num_proc):
        edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
        mean_ed += float(edit_dist)
        mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
      num_left -= num_proc
    mean_norm_ed = mean_norm_ed / num
    mean_ed = mean_ed / num
    print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
          % (num, mean_ed, mean_norm_ed))

  def on_epoch_end(self, epoch, logs={}):
    self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
    self.show_edit_distance(256)
    word_batch = next(self.text_img_gen)[0]
    res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
    if word_batch['the_input'][0].shape[0] < 256:
      cols = 2
    else:
      cols = 1
    for i in range(self.num_display_words):
      pylab.subplot(self.num_display_words // cols, cols, i + 1)
      if K.image_dim_ordering() == 'th':
        the_input = word_batch['the_input'][i, 0, :, :]
      else:
        the_input = word_batch['the_input'][i, :, :, 0]
      pylab.imshow(the_input.T, cmap='Greys_r')
      pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (translate_to_geo(word_batch['source_str'][i], res[i])))
    fig = pylab.gcf()
    fig.set_size_inches(10, 13)
    pylab.savefig(os.path.join(self.img_dir, 'e%02d.png' % (epoch)))
    pylab.close()

