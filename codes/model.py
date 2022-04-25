# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:07:40 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import tensorflow as tf  
import numpy as np
from fusion import Fusion  

class Model(object):
    
    def __init__(self):
        self.fusion = Fusion() 
        self.loaded_model = self.load_model()
    
    def load_model(self):
        model = tf.keras.models.load_model('../model/seq-10-32-32/weights-improvement-01-0.0000-bigger.hdf5')
        return model
    
    def get_results(self, results):
        final = {}
        for key, val in results.items():
            best_acc = 0
            label = ''
            for x, y in val.items(): 
                if best_acc < y['acc']:
                    best_acc = y['acc']
                    label = x
            final[key] = label
        return final
    
    def make_prediction(self, dict_users): 
        
        results = {}   
            
        for key, val in dict_users.items():
            print('Processing user', key)
            if key not in results:
                results[key] = {}
                
            for k, v in val.items():
                
                print('Now at >', k)
                
                if k not in results[key]:
                    results[key][k] = {}
                    
                if 'pred' not in results[key][k]:
                    results[key][k]['pred'] = [] 
                
                # Make a prediction per sequence
                for i, o in v: # ins, ops
                    
                    for x, y in zip(i, o):  
                        print('Converting data to image')
                        to_check = np.expand_dims(self.fusion.insole_openpose_to_img(x, y), axis=0)
                        
                        print('Make a prediction')
                        pred = np.squeeze(self.loaded_model.predict(to_check)) # 0 = PAIRED; 1 = UNPAIRED 
                        
                        if pred[0] > pred[1]:
                            results[key][k]['pred'].append(0)
                        else:
                            results[key][k]['pred'].append(1) 
                            
                results[key][k]['acc'] = (len(results[key][k]['pred']) - np.count_nonzero(np.array(results[key][k]['pred']))) / len(results[key][k]['pred'])
             
        final = self.get_results(results)
        
        return final, results