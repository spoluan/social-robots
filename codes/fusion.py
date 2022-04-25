# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 00:59:12 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import pandas as pd
import numpy as np
from drawinsole import DrawInsole
from drawskeleton import DrawSkeleton
from imgshow import ImageShow

class Fusion(object):
    
    def __init__(self):  
        self.drawinsole = DrawInsole() 
        self.drawskeleton = DrawSkeleton()
    
    def fusion(self, downsampled_openpose, downsampled_insole_data):
        # Get indexes from insole and openspose if they have the same time
        insoletime = downsampled_insole_data.TIME
        openposetime = pd.Series(list(map(lambda x: ':' . join(x.split(':')[:-1]), downsampled_openpose.TIME)))
        
        insole_indexes = []
        openpose_indexes = []
        for time in np.unique(openposetime):
            insole_records = np.squeeze(np.where(insoletime == time)) 
             
            if len(insole_records) == 5:
                openpose_records = np.squeeze(np.where(openposetime == time)) 
                
                insole_indexes.extend(insole_records) 
                openpose_indexes.extend(openpose_records) 
                
        fusion_downsampled_insole_data = pd.DataFrame(np.array(downsampled_insole_data)[insole_indexes], columns=downsampled_insole_data.columns)
        fusion_downsampled_openpose = pd.DataFrame(np.array(downsampled_openpose)[openpose_indexes], columns=downsampled_openpose.columns)
        
        return fusion_downsampled_insole_data, fusion_downsampled_openpose
    
    def insole_openpose_to_img(self, downsampled_insole_data, downsampled_openpose): 
    
        fusion_downsampled_insole_data, fusion_downsampled_openpose = downsampled_insole_data, downsampled_openpose #  self.fusion(downsampled_openpose, downsampled_insole_data)
        
        break_at = 10 # 10 is frequency of every 1 second
        to_check = []
        for insole, openpose in zip(fusion_downsampled_insole_data.itertuples(), fusion_downsampled_openpose.itertuples()):
            r_pressure_data = self.drawinsole.run("RIGHT", [insole.R_THUMB, insole.R_INNER_BALL, insole.R_OUTER_BALL, insole.R_HEEL])
            l_pressure_data = self.drawinsole.run("LEFT", [insole.L_THUMB, insole.L_INNER_BALL, insole.L_OUTER_BALL, insole.L_HEEL])
            
            skeletal = self.drawskeleton.run([openpose.HIP_RIGHT_X, openpose.HIP_RIGHT_Y, openpose.HIP_LEFT_X, openpose.HIP_LEFT_Y, openpose.KNEE_RIGHT_X, openpose.KNEE_RIGHT_Y, 
                                         openpose.KNEE_LEFT_X, openpose.KNEE_LEFT_Y, openpose.ANKLE_RIGHT_X, openpose.ANKLE_RIGHT_Y, openpose.ANKLE_LEFT_X, openpose.ANKLE_LEFT_Y])
            
            # ImageShow(skeletal)
            
            abc = np.concatenate([r_pressure_data, l_pressure_data, skeletal], axis=-1) # abc.shape 
            
            to_check.append(abc)
            
            if break_at == 1:
                break
            break_at -= 1
            
        return np.array(to_check)
    
    def get_starting_time(self, insoles, openposes):
        
        # Time checking
        times = []
        for t in insoles:
            times.append(t.TIME)
        for t in openposes:
            times.append(t.TIME)
            
        starting_time = ''
        for x in times[0]:
            count = 0
            for y in times[1:]:
                if x in list(y):
                    count += 1 
            if count > len(times[1:]) - 1:
                starting_time = x
                break
            
        return starting_time
    
    def equalize_the_staring_time(self, insoles, openposes):
        
        starting_time = self.get_starting_time(insoles, openposes)
    
        # Get starting index based on the starting time
        for x in range(len(insoles)):
            get = np.squeeze(np.where(np.array(insoles[x].TIME) == starting_time))[0]
            insoles[x] = pd.DataFrame(np.array(insoles[x])[np.arange(get, len(insoles[x]) - 1)], columns=insoles[x].columns)
        for x in range(len(openposes)):
            get = np.squeeze(np.where(np.array(openposes[x].TIME) == starting_time))[0]
            openposes[x] = pd.DataFrame(np.array(openposes[x])[np.arange(get, len(openposes[x]) - 1)], columns=openposes[x].columns)
        
        return insoles, openposes
    
    def get_sequence_data(self, insoles, openposes):
        # Get the sequence data
        seq = np.arange(0, 10) + np.expand_dims(np.arange(0, 250, 10), axis=1)
        ins_seqs = []
        for x in insoles:
            xy = []
            for y in np.array(x)[seq]:
                xy.append(pd.DataFrame(y, columns=x.columns))
            ins_seqs.append(xy)
        
        ops_seqs = []
        for x in openposes:
            xy = []
            for y in np.array(x)[seq]:
                xy.append(pd.DataFrame(y, columns=x.columns))
            ops_seqs.append(xy)
        return ins_seqs, ops_seqs
    
    def user_fusion(self, ins_seqs, ops_seqs):
        dict_users = {}
        label = []
        for check_ins in ins_seqs: 
            label = check_ins[0].NAME[0]
            if label not in dict_users:
                dict_users[label] = {}
            for check_ops in ops_seqs:
                label_ops = str(check_ops[0].NAME[0])
                if label + label_ops not in dict_users[label]:
                    dict_users[label][label + label_ops] = []
                dict_users[label][label + label_ops].append([check_ins, check_ops])
                
        return dict_users