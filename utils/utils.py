#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage


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
		result += chars[char] if char in chars else char
	return result


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
