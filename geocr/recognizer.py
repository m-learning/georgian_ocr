"""
python -m cnn.ocr.recognizer --image path_to_image --model weights.h5 [--width 512]

python -m geocr/recognizer --weights results/weights20.h5 --image data/test.jpeg --width 512 --model results/model256.yaml

image height must be 64 px, width 128 or 512 px (128 is default)
"""

from __future__ import unicode_literals

import Image
import argparse
import itertools

from keras import backend as K
from keras.models import model_from_yaml

import numpy as np


# import geocr.network_model as network
# import geocr.training_flags as flags
chars = {
      u'ა': 'a',
      u'ბ': 'b',
      u'გ': 'g',
      u'დ': 'd',
      u'ე': 'e',
      u'ვ': 'v',
      u'ზ': 'z',
      u'თ': 'T',
      u'ი': 'i',
      u'კ': 'k',
      u'ლ': 'l',
      u'მ': 'm',
      u'ნ': 'n',
      u'ო': 'o',
      u'პ': 'p',
      u'ჟ': 'J',
      u'რ': 'r',
      u'ს': 's',
      u'ტ': 't',
      u'უ': 'u',
      u'ფ': 'f',
      u'ქ': 'q',
      u'ღ': 'R',
      u'ყ': 'y',
      u'შ': 'S',
      u'ჩ': 'C',
      u'ც': 'c',
      u'ძ': 'Z',
      u'წ': 'w',
      u'ჭ': 'W',
      u'ხ': 'x',
      u'ჯ': 'j',
      u'ჰ': 'h',
      u' ': ' ',
      }


def translate(text):
	result = ''
	for char in text:
		result += chars[char]
	return result


# from keras.optimizers import SGD
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
		ret.append(translate(out_str))
	return ret


def init_arguments():
	
	parser = argparse.ArgumentParser(description='Georgian OCR')
	parser.add_argument('--image', metavar='image_path', type=str,
						help='Path to the image to recognize.')
	parser.add_argument('--weights', metavar='weights_path', type=str,
						help='Path to the weights.')
	parser.add_argument('--width', metavar='image_width', type=int,
						help='image width: 128 or 512 (128 is default)', default=128)
	parser.add_argument('--model', metavar='model', type=str,
						help='Path to model')
	return parser.parse_args()

def _config_array(img_w, img_h, array):
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
	print(array.shape)
	
	return array

if __name__ == '__main__':
	"""Runs OCR recognizer"""
	
	args = init_arguments()
	img_w = args.width
	img_h = 64
	
	img = Image.open(args.image)
	img = img.convert("L")
	array = np.asarray(img.getdata(), dtype=np.float32)
	array /= 255.0
	array = np.expand_dims(array, 0)
	
	array = _config_array(img_w, img_h)
	
	yaml_file = open(args.model, 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	model = model_from_yaml(loaded_model_yaml)
	
	model.load_weights(args.weights)
	model.summary()
	print("Loaded model from disk")
	
	# sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
	# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	pred = model.predict(array, batch_size=1, verbose=1)
	
	print(decode_result(pred))
