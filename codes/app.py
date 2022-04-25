# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 01:18:44 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
""" 

# global_address="D:\Journal - 1\Implementation\codes"   
import os
# os.chdir(global_address)  
import pandas as pd
from downsampling import DownSampling 
from fusion import Fusion  
from model import Model
 
downsampling = DownSampling()
fusion = Fusion() 
model = Model()
  
def get_insoles():
    insole_data = pd.read_csv('../Sample-test-data/smart-insole.csv')
    downsampled_insole_data = downsampling.downsample(insole_data)
    return downsampled_insole_data
    
def get_openpose():
    filter_columns = 'HIP_RIGHT_X	HIP_RIGHT_Y	HIP_LEFT_X	HIP_LEFT_Y	KNEE_RIGHT_X	 KNEE_RIGHT_Y	 KNEE_LEFT_X	 KNEE_LEFT_Y	 ANKLE_RIGHT_X	ANKLE_RIGHT_Y	ANKLE_LEFT_X	 ANKLE_LEFT_Y TIME' . split()
    openpose_data = pd.read_csv('../Sample-test-data/open-pose.csv')[filter_columns]
    downsampled_openpose = downsampling.downsample(openpose_data)
    return downsampled_openpose   

def manage_fusion():
    src_insoles = ['../Sample-test-data/smart-insole-A.csv', '../Sample-test-data/smart-insole-E.csv']
    src_openpose = ['../Sample-test-data/open-pose-1.csv', '../Sample-test-data/open-pose-2.csv']
      
    # Load the data and downsample to 5 samples per second
    insoles = list(map(lambda x: downsampling.downsample(pd.read_csv(x)), src_insoles))
    openposes = list(map(lambda x: downsampling.downsample(pd.read_csv(x)), src_openpose))
     
    # Equalize the staring time
    insoles, openposes = fusion.equalize_the_staring_time(insoles, openposes)
    ins_seqs, ops_seqs = fusion.get_sequence_data(insoles, openposes)
    
    # Fuse preparing to predict 
    dict_users = fusion.user_fusion(ins_seqs, ops_seqs)
     
    # Make a prediction
    results, prob_results = model.make_prediction(dict_users)
 
    print(results)


if __name__  == '__main__':
    manage_fusion()

        