#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mar 11, 2017

Translates characters between languages

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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

def translate_to_geo(text):
  """Translates text per characters
    Args:
      text - input text
    Returns:
      result - translated text
  """
  
  result = ''
  
  for char in text:
    result += chars[char]
  
  return result
