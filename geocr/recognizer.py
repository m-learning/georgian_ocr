"""
python -m cnn.ocr.recognizer --image path_to_image --model weights.h5 [--width 512]

python -m cnn.ocr.recognizer --weights ./datas/ocr/weights24.h5 --image ./datas/ocr/test.jpeg --width 512

image height must be 64 px, width 128 or 512 px (128 is default)
"""

import Image
import argparse
import itertools

from keras import backend as K
#from keras.optimizers import SGD

import geocr.network_model as m
import numpy as np


def decode_result(out):
	ret = []
	for j in range(out.shape[0]):
		out_best = list(np.argmax(out[j, 2:], 1))
		out_best = [k for k, _ in itertools.groupby(out_best)]
		# 26 is space, 27 is CTC blank char
		out_str = ''
		for c in out_best:
			if 0 <= c < 26:
				out_str += chr(c + ord('a'))
			elif c == 26:
				out_str += ' '
		ret.append(out_str)
	return ret


def init_arguments():
	parser = argparse.ArgumentParser(description='Georgian OCR')
	parser.add_argument('--image', metavar='image_path', type=str,
						help='Path to the image to recognize.')
	parser.add_argument('--weights', metavar='weights_path', type=str,
						help='Path to the weights.')
	parser.add_argument('--width', metavar='image_width', type=int,
						help='image width: 128 or 512 (128 is default)', default=128)
	return parser.parse_args()


args = init_arguments()
img_w = args.width
img_h = 64

img = Image.open(args.image)
img = img.convert("L")
array = np.asarray(img.getdata(), dtype=np.float32)
array /= 255.0
array = np.expand_dims(array, 0)

if K.image_dim_ordering() == 'th':
	array = array.reshape([1, 1, img_w, img_h])
else:
	array = array.reshape([1, img_w, img_h, 1])
print(array.shape)

(_, input_data, y_pred, model) = m.ocr_network(img_w)
model.summary()
model.load_weights(args.weights)
model.summary()
print("Loaded model from disk")

# sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
pred = model.predict(array, batch_size=1, verbose=1)

print(decode_result(pred))
