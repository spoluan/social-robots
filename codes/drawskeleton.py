# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 00:03:37 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import cv2 # pip install opencv-contrib-python
import numpy as np
from skimage.transform import resize # pip install scikit-image

class DrawSkeleton(object): 
        
    def round_int(self, number): 
        x = 0
        try:
            if number == float("inf") or number == float("-inf"):
                return float("nan")
            else:
                x = int(round(number))
        except:
            return float("nan") 
        return x

    def draw_body_bone(self, joint_x1, joint_y1, joint_x2, joint_y2): 
      
        # x_based = self.width / self.ori_width_kinect
        # y_based = self.height / self.ori_height_kinect
        
        #(4) * 5 / 6
        #(3.3333333333333335) * 6 / 5 ---> x = open_pose * 1920 / 1080
        #                             ---> y = open_pose * 1080 / 720
        
        # start_x =  (self.height - self.round_int(self.round_int(joint_x1)  * x_based)) # Right
        # start_y =  (self.round_int(self.round_int(joint_y1) * y_based))
        # end_x =  (self.height - self.round_int(self.round_int(joint_x2) * x_based)) # Right
        # end_y =  (self.round_int(self.round_int(joint_y2) * y_based)) 
        
        start_x = self.round_int(self.round_int(joint_x1))
        start_y = self.round_int(self.round_int(joint_y1))
        end_x = self.round_int(self.round_int(joint_x2))
        end_y = self.round_int(self.round_int(joint_y2)) 
      
        try:   
            cv2.line(self.color_frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2) # thickness
        except:
            pass 
        
    def run(self, coordinate, dim=32): 
        
        hip_right_x = int(coordinate[0])
        hip_right_y = int(coordinate[1])
        hip_left_x = int(coordinate[2])
        hip_left_y = int(coordinate[3])
        knee_right_x = int(coordinate[4])
        knee_right_y =int(coordinate[5])
        knee_left_x = int(coordinate[6])
        knee_left_y = int(coordinate[7])
        ankle_right_x = int(coordinate[8])
        ankle_right_y = int(coordinate[9])
        ankle_left_x = int(coordinate[10])
        ankle_left_y = int(coordinate[11]) 
        
        shifting = ankle_right_x # Shifting the object close to the 0 x
        gap = 50 # Make a gap to the object
        
        hip_right_x = hip_right_x - shifting + gap
        hip_right_y = coordinate[1] 
        hip_left_x = hip_left_x - shifting + gap 
        hip_left_y = coordinate[3] 
        knee_right_x = knee_right_x - shifting + gap
        knee_right_y = coordinate[5] 
        knee_left_x = knee_left_x - shifting + gap 
        knee_left_y = coordinate[7] 
        ankle_right_x = ankle_right_x - shifting + gap 
        ankle_right_y = coordinate[9]
        ankle_left_x = ankle_left_x - shifting + gap 
        ankle_left_y = coordinate [11]  
        
        self.width = ankle_left_x + gap
        self.height = int(ankle_right_x + gap) if ankle_right_x > ankle_right_x else int(ankle_right_y + gap)
          
        self.color_frame = np.ones((self.height, self.width)) * 255 
        
        self.draw_body_bone(hip_right_x, hip_right_y, knee_right_x, knee_right_y) 
        self.draw_body_bone(hip_left_x, hip_left_y, knee_left_x, knee_left_y) 
        self.draw_body_bone(knee_right_x, knee_right_y, ankle_right_x, ankle_right_y) 
        self.draw_body_bone(knee_left_x, knee_left_y, ankle_left_x, ankle_left_y) 
        self.draw_body_bone(hip_right_x, hip_right_y, hip_left_x, hip_left_y) 
        
        skeletal = self.color_frame[hip_right_y - gap:, :]
         
        skeletal = resize(np.array(skeletal), (dim, dim, 1))
        
        return skeletal