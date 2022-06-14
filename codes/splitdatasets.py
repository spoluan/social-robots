# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 22:24:16 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import pickle 
import numpy as np
import os
import uuid 

addr = 'D:\\Journal - 1\\Implementation'

class PickleDumpLoad(object):   
    
    def save_config(self, obj, addr):  
        with open(addr, 'wb') as config_f: 
            pickle.dump(obj, config_f)    
        print('{} saved!' . format(addr))
        
    def load_config(self, addr):  
        with open('{}' . format(addr), 'rb') as f_in:
             obj = pickle.load(f_in)
        return obj   

class GenerateNewDataset(object):
    
    def load_data(self): 
        print('Load dataset.')
        p_load = PickleDumpLoad() 
        loaded_data = p_load.load_config("../train-sets/normal/seq-10-32-32/train-dataset.pkl") 
        loaded_data_flipped = p_load.load_config('../train-sets/flipped/seq-10-32-32/train-dataset.pkl')
     
        for key, val in loaded_data.items(): 
            loaded_data[key].extend(loaded_data_flipped[key])  
        
        return loaded_data
    
    def prepare_folder_for_dataset_generation(self, loaded_data):
        print('Prepare folders to save.')
        labels = np.unique(np.array(loaded_data['LABEL']))
        for label in labels:
            folder_to_create = os.path.join(addr, 'new_train_sets', label)
            if not os.path.exists(folder_to_create):
                os.makedirs(folder_to_create)
                 
    def save_datasets(self, loaded_data):
        print('Start save the datasets.')
        save = PickleDumpLoad()
        for count, (data, label) in enumerate(zip(loaded_data['DATA'], loaded_data['LABEL'])): 
            save.save_config(data, os.path.join('../new_train_sets', label, str(uuid.uuid4()))) 
            print(f"Finish {count} from {len(loaded_data['LABEL'])}.")
    
    def main(self):
        loaded_data = self.load_data()
        self.prepare_folder_for_dataset_generation(loaded_data)
        self.save_datasets(loaded_data)
        
app = GenerateNewDataset()
app.main()
        