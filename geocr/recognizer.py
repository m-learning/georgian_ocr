#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import Image
import argparse
import itertools
import os

import numpy as np
from keras import backend as K
from keras.models import model_from_yaml

from utils.utils import translate

SEPARATOR = '\n'

def decode_result(out):
	ret = []
	for j in range(out.shape[0]):
		out_best = list(np.argmax(out[j, 2:], 1))
		out_best = [k for k, _ in itertools.groupby(out_best)]
		# 33 is space, 34 is CTC blank char
		out_str = ''
		for c in out_best:
			if 0 <= c < 33:
				out_str += unichr(c + ord(u'ა'))
			elif c == 33:
				out_str += ' '
		ret.append(out_str)
	return ret[0] + SEPARATOR


def init_arguments():
	
	parser = argparse.ArgumentParser(description='Georgian OCR')
	parser.add_argument('-i', '--image', metavar='image_path', type=str,
						help='Path to the image to recognize.')
	parser.add_argument('-W', '--weights', metavar='weights_path', type=str,
						help='Path to the weights.')
	parser.add_argument('-w', '--width', metavar='image_width', type=int,
						help='image width: 128 / 256 / 512 (256 is default)', default=256)
	parser.add_argument('-m', '--model', metavar='model', type=str,
						help='Path to model')
	parser.add_argument('-e', '--english', action='store_true',
						help='print output in english letters')
	return parser.parse_args()


def _config_array(array, img_w, img_h):
	"""Configuration for image arrays
		Args:
			img_w - image width
			img_h - image height
			array - input image tensor
		Returns:
			array - configured image tensor
	"""
	
	if K.image_dim_ordering() == 'th':
		array = array.reshape([1, 1, img_w, img_h])
	else:
		array = array.reshape([1, img_w, img_h, 1])
	# print(array.shape)
	return array

def predict_text(model, image):
	img = Image.open(image)
	img = img.convert("L")
	array = np.asarray(img.getdata(), dtype=np.float32)
	array /= 255.0
	array = np.expand_dims(array, 0)

	array = _config_array(array, img_w, img_h)
	pred = model.predict(array, batch_size=1, verbose=0)
	
	return decode_result(pred)

if __name__ == '__main__':
	"""Runs OCR recognizer"""

	args = init_arguments()
	img_w = args.width
	img_h = 64

	yaml_file = open(args.model, 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	model = model_from_yaml(loaded_model_yaml)

	model.load_weights(args.weights)
	# model.summary()
	result = ''

	if os.path.isfile(args.image):
		result += predict_text(model, args.image)
	elif os.path.isdir(args.image):
		for (dir_path, dir_names, file_names) in os.walk(args.image):
			file_names = sorted(file_names)
			for file_name in file_names:
				result += predict_text(model, os.path.join(dir_path, file_name))
			break
	else:
		raise IOError('%s is not a file' % args.image)

	if args.english:
		print(translate(result))
	else:
		print(result)
