#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import matplotlib.image as mpimg
import os

import cairocffi as cairo
import numpy as np
from keras.preprocessing import image

from utils.utils import translate, speckle

OUTPUT_DIR = 'data/chunks'


def paint_text(text, w, h, rotate=False, ud=False, multi_fonts=False):
	surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
	with cairo.Context(surface) as context:
		context.set_source_rgb(1, 1, 1)  # White
		context.paint()
		# this font list works in Centos 7
		if multi_fonts:
			fonts = ['AcadNusx', 'AcadMtavr', 'Acad Nusx Geo', 'LitNusx', 'Chveulebrivi TT', 'DumbaNusx']
			context.select_font_face(np.random.choice(fonts), cairo.FONT_SLANT_NORMAL,
									 np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
		else:
			context.select_font_face('AcadNusx', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
		context.set_font_size(25)
		box = context.text_extents(translate(text))
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
		context.show_text(translate(text))

	buf = surface.get_data()
	a = np.frombuffer(buf, np.uint8)
	a.shape = (h, w, 4)
	# a = a[:, :, 0]  # grab single channel
	a = a.astype(np.float32) / 255.0
	a = np.expand_dims(a, 0)
	if rotate:
		a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
	a = speckle(a)
	mpimg.imsave(os.path.join(OUTPUT_DIR, text + ".jpg"), a[0])
	return a


def init_arguments():
	parser = argparse.ArgumentParser(description='random image generator')
	parser.add_argument('text', metavar='text', type=str, nargs='+',
						help='text to generate.')
	parser.add_argument('-w', '--width', metavar='image_width', type=int,
						help='image width: 128 / 256 / 512 (256 is default)', default=256)
	parser.add_argument('--height', metavar='image_height', type=int,
						help='image width (64 is default)', default=64)
	parser.add_argument('-s', '--save_path', metavar='save_path', type=str, default='data/chunks',
						help='path to save generated images')
	return parser.parse_args()


if __name__ == '__main__':
	args = init_arguments()
	OUTPUT_DIR = args.save_path
	for word in args.text:
		paint_text(word.decode('utf-8'), args.width, args.height, False, True, True)

