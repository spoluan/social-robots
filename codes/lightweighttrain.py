 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:17:13 2020 MSSLAB CSE YZU TAIWAN

@author: SEVENDI ELDRIGE RIFKI POLUAN
""" 
  
import numpy as np  
from tensorflow.keras.layers import Input, Dense, Flatten, TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras import Model as ModelKeras 
from tensorflow.keras.applications import VGG16
from datagenerator import DataGenerator 
import os 
 
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
        self.EPOCH = 1
        self.BATCH_SIZE = 4
    
    def prepare_dataset_paths(self):
        addr = '../new_train_sets'
        paths = []
        label = []
        for root, dirs, files in os.walk(addr):
            for file in files:
                path = os.path.join(root, file)
                if 'UNPAIRED' in path:
                    label.append('UNPAIRED')
                else:
                    label.append('PAIRED')
                paths.append(path) 
        return paths, label
    
    def train(self, x_y_train): 
        model = CNN_Model().transfer_learning(dimension=32, seq=10)
         
        from tensorflow.keras.callbacks import ModelCheckpoint
        
        save = 'weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5'
        filepath = os.path.join("../model/seq-10-32-32", save)
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]   
        
        model.fit(x_y_train, batch_size=self.BATCH_SIZE, verbose=1, epochs=self.EPOCH, callbacks=callbacks_list)
            
    def main(self):
        
        # LOAD DATASET PATHS AND LABELS
        x_paths, label = self.prepare_dataset_paths() 
        
        # LABEL ENCODER
        x_label = EncodeLabelsCategorically().manual_categorical_labeling(label)
         
        # PUT INTO THE DATA GENERATOR 
        x_y_train = DataGenerator(
            x_paths,  
            x_label,
            batch_size=self.BATCH_SIZE) 
 
        # TRAIN THE MODEL
        self.train(x_y_train)

app = Train()
app.main()
 
