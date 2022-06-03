 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:17:13 2020 MSSLAB CSE YZU TAIWAN

@author: SEVENDI ELDRIGE RIFKI POLUAN
""" 

import pickle as pickle   
import numpy as np  
from tensorflow.keras.layers import Input, Dense, Flatten, TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras import Model as ModelKeras 
from tensorflow.keras.applications import VGG16
import os

class PickleDumpLoad(object):   
    
    def save_config(self, obj, addr):  
        with open(addr, 'wb') as config_f: 
            pickle.dump(obj, config_f)    
        print('{} saved!' . format(addr))
        
    def load_config(self, addr):  
        with open('{}' . format(addr), 'rb') as f_in:
             obj = pickle.load(f_in)
        return obj   

class CNN_Model(object):   
    
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
    
# LABEL ENCODING CLASS 
class EncodeLabelsCategorically(object):  
    
    def manual_categorical_labeling(self, label_to_encode=[], num_classes=2):
        uniqueness = sorted(list(set(label_to_encode)))
        labs = np.zeros((len(label_to_encode), num_classes))
        for ind, val in enumerate(label_to_encode): 
            labs[ind][uniqueness.index(val)] = 1 
        return labs 
 
class Train(object):
    def __init__(self):
        pass
    
    def load_dataset(self):
        
        # CREATE OBJECT TO USE PICKLE
        p_load = PickleDumpLoad() 
        loaded_data = p_load.load_config("../train-sets/normal/seq-10-32-32/train-dataset.pkl") 
        loaded_data_flipped = p_load.load_config('../train-sets/flipped/seq-10-32-32/train-dataset.pkl')
        
        for key, val in loaded_data.items():
            loaded_data[key].extend(loaded_data_flipped[key])  
         
        # SHUFFLE THE DATA 
        from sklearn.utils import shuffle 
        id_random = shuffle(np.arange(len(np.array(loaded_data_flipped['LABEL']))))
        
        x_train = np.array(loaded_data_flipped['DATA'])[id_random]  
        y_train =  np.array(loaded_data_flipped['LABEL'])[id_random]  
          
        y_train_encoded_categorically = EncodeLabelsCategorically().manual_categorical_labeling(np.array(y_train).reshape(np.array(y_train).shape[0]))
          
        print('# TRAINING')
        for labs in list(set(np.argmax(y_train_encoded_categorically, axis=1))):
            print(labs, len(np.argmax(y_train_encoded_categorically, axis=1)[np.array(np.argmax(y_train_encoded_categorically, axis=1) == labs)])) 
         
        y_train = y_train_encoded_categorically
        
        return x_train, y_train
    
    def train(self, x_train, y_train):
         
        model = CNN_Model().transfer_learning(dimension=32, seq=10)
        
        from tensorflow.keras.callbacks import ModelCheckpoint
        save = 'weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5'
        filepath = os.join.path("../model/seq-10-32-32", save)
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]  
        
        epochs = 1
        for ep in range(epochs):
            print('epoch', ep, 'from', epochs)
            model.fit(x_train, y_train, batch_size=128, verbose=1, epochs=1, callbacks=callbacks_list)
            
    def main(self):
        
        x_train, y_train = self.load_dataset()
        self.train(x_train, y_train)

app = Train()
app.main()