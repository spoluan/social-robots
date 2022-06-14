# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 22:20:53 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import numpy as np
import tensorflow as tf 
import pickle

class PickleDumpLoad(object):   
    
    def save_config(self, obj, addr):  
        with open(addr, 'wb') as config_f: 
            pickle.dump(obj, config_f)    
        print('{} saved!' . format(addr))
        
    def load_config(self, addr):  
        with open('{}' . format(addr), 'rb') as f_in:
             obj = pickle.load(f_in)
        return obj   

load = PickleDumpLoad()

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, x_train, y_train, batch_size=4, shuffle=True, frame_length=20):
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.shuffle = shuffle
        self.frame_length = frame_length
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.y_train) / self.batch_size))
    
    def __getitem__(self, index): 
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]     
        
        x_train = []
        for path in np.array(self.x_train)[indexes]: 
            data = self.data_load(path)  
            x_train.append(data)
         
        x_train = np.array(x_train)  
        y_train = np.array(self.y_train)[indexes]   
         
        return x_train, y_train
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)    
     
    def data_load(self, path):
        data = np.array(load.load_config(path))
        return data 
