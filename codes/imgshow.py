# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 00:06:24 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import matplotlib.pyplot as plt
import numpy as np

class ImageShow(object): 
    def __init__(self, image_to_show):  
        plt.imshow(np.squeeze(image_to_show)) #, cmap='gray')
        plt.show()