# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 00:05:14 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import numpy as np
import pandas as pd

class DownSampling(object):
    
    def downsample(self, data): 
        
        times = data['TIME']
        
        indexes = []
        current = ''
        temp = []
        for no, time in enumerate(times):
            try:
                time = time.split(':')[-1] if len(time.split(':')) < 4 else time.split(':')[-2]
                current = time if current == '' else current
                if time in current:
                    temp.append(no)
                else:
                    if len(temp) > 5: # Downsample to 5 samples per second
                        temp = temp[-5:] # Take only last 5 samples 
                        indexes.extend(temp)
                    current = time
                    temp = []
            except:
                pass
        np_data = np.array(data)[np.array(indexes)]
        pd_data = pd.DataFrame(np_data, columns=data.columns)
        
        # Remove miliseconds
        pd_data.TIME = [x if len(x.split(':')) < 4 else ':' . join(x.split(':')[:-1]) for x in pd_data.TIME]
        
        return pd_data