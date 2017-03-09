# coding=utf-8

import unittest
import sys
sys.path.append(".")
import os
import re
from image_ocr import *

class main(unittest.TestCase):

    def createResponse(self, path):
      curr = os.path.dirname(os.path.realpath(__file__))
      path =  os.path.join(curr, path)
      with file(path) as f:
        return HtmlResponse(url=path, body = f.read())
      
    ''' Test points system 
    '''
    def test_text_to_labels (self):
      print text_to_labels(u'ადამიანი', 28)
