# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:07:40 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import tensorflow as tf  
import numpy as np
from fusion import Fusion  
import glob
import os
from tensorflow.keras.layers import Input, Dense, Flatten, TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras import Model as ModelKeras 
from tensorflow.keras.applications import VGG16

class Model(object):
    
    def __init__(self):
        self.fusion = Fusion() 
        self.loaded_model = self.load_model()

    def transfer_learning(self, dimension=32, seq=4): 
        
        def vgg16(layer_name='sk'):
            vgg_model = VGG16(weights="imagenet", include_top=False) # 224
            seq = Sequential()
            for x in range(len(vgg_model.layers)):
                layer = vgg_model.layers[x]
                layer._name = 'l_{}_{}' . format(layer_name, x)
                layer.trainable = False
                seq.add(layer)
            return seq # seq.summary() # vgg_model.summary()
       
        input_shape = Input(shape=(seq, dimension, dimension, 3))  
        x = TimeDistributed(vgg16(layer_name='sk'))(input_shape) 
        x = TimeDistributed(Flatten())(x) 
        x = Dense(512, activation='relu')(x) 
        x = LSTM(512, return_sequences=False)(x)  
        x = Dense(512, activation='relu')(x)  
        x = Dense(2, activation='softmax', name='predictions')(x) 
        model = ModelKeras(inputs=input_shape, outputs=x)
        op = optimizers.Adam(lr=0.0001) 
        model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])
         
        model.summary() 
        return model 
    
    def load_model_v1(self):
        files = glob.glob('..\model\seq-10-32-32/*')
        file_model = max(files, key=os.path.getctime)
        model =  tf.keras.models.load_model(file_model)
        return model

    def load_model(self):
        files = glob.glob('..\model\seq-10-32-32/*')
        file_model = max(files, key=os.path.getctime)
        model = self.transfer_learning(dimension=32, seq=10)
        model.load_weights(file_model)
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