# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 00:00:48 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
""" 

import pandas as pd
import math 
import seaborn as sns # pip install seaborn
import numpy as np 
import sys
from tkinter import * # sudo apt-get install python3-tk
from skimage.transform import resize # pip install scikit-image

class DrawInsole(object):
    
    def __init__(self):  

        self.left_shoe_base_mask = [
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111222111111111111"),
                        list("111111111111111111112222211111111111"),
                        list("111111111111111111112222211111111111"),
                        list("111111111111111111111222111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111333111111111"),
                        list("111111111111111111111113333311111111"),
                        list("111111111444111111111113333311111111"),
                        list("111111114444411111111111333331111111"),
                        list("111111144444111111111111133331111111"),
                        list("111111444441111111111111111111111111"),
                        list("111111144411111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111115555555111111111"),
                        list("111111111111111111155555555511111111"),
                        list("111111111111111111155555555511111111"),
                        list("111111111111111111115555555111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111"),
                        list("111111111111111111111111111111111111")]

        self.right_shoe_base_mask = [
                list("111111111111111111111111111111111111"),
                list("111111111111222111111111111111111111"),
                list("111111111112222211111111111111111111"),
                list("111111111112222211111111111111111111"),
                list("111111111111222111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111333111111111111111111111111"),
                list("111111113333311111111111111111111111"),
                list("111111113333311111111111444111111111"),
                list("111111133333111111111114444411111111"),
                list("111111133331111111111111444441111111"),
                list("111111111111111111111111144444111111"),
                list("111111111111111111111111114441111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111555555511111111111111111111"),
                list("111111115555555551111111111111111111"),
                list("111111115555555551111111111111111111"),
                list("111111111555555511111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111"),
                list("111111111111111111111111111111111111")] 

        self.MAX_PRESSURE = 80 # default 80 
        self.shoe_height = 36
        self.shoe_width = 40     

    def get_sensor_area_list(self, pressure_value): # pressure value range --> 2, 3, 4, 5
        sensor_area = list()
        for x in range(self.shoe_width):
            for y in range(self.shoe_height):  
                try:
                    if int(self.status_shoe_base_mask[x][y]) == pressure_value:
                        sensor_area.append([x, y])
                except:
                    pass
        return sensor_area

    def get_point_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    def get_distance_array(self, sensor_area_list):
        distances = np.empty((self.shoe_width, self.shoe_height), dtype=float)
        for x in range(self.shoe_width):
            for y in range(self.shoe_height):
                distances[x][y] = sys.float_info.max # set a default value
                for sp in sensor_area_list: 
                    if (self.get_point_distance(x, y, sp[0], sp[1]) < distances[x][y]):
                        distances[x][y] = self.get_point_distance(x, y, sp[0], sp[1]) 
        return distances    
     
    def get_grid_pressure(self, pressure_data, x, y):  
        pa = int(pressure_data[0] / (self.distance_a[x][y] / 2 + 1)) # thumb
        pb = int(pressure_data[1] / (self.distance_b[x][y] / 2 + 1)) # inner_ball
        pc = int(pressure_data[2] / (self.distance_c[x][y] / 2 + 1)) # outer_ball
        pd = int(pressure_data[3] / (self.distance_d[x][y] / 2 + 1)) # heel
        d_sum = min(pa + pb + pc + pd, self.MAX_PRESSURE)
        return int(math.sin(d_sum / self.MAX_PRESSURE * math.pi / 2) * self.MAX_PRESSURE)
 
    def draw_grid_color(self, pressure_data): 
        grid_mask = np.empty((self.shoe_width, self.shoe_height), dtype=float) 
        for x in range(self.shoe_width):
            for y in range(self.shoe_height):  
                grid_mask[x][y] = self.get_grid_pressure(pressure_data, x, y)   
        df_grid_mask = pd.DataFrame(grid_mask) 
        # ax = sns.heatmap(df_grid_mask, xticklabels=False, yticklabels=False, vmin=0, vmax=80, annot=True, cbar=False) 
        return df_grid_mask # ax   

    def run(self, status, pressure_data, dim=32):  
        
        if status == "LEFT":  
            self.status_shoe_base_mask = self.left_shoe_base_mask 
        if status == "RIGHT": 
            self.status_shoe_base_mask = self.right_shoe_base_mask 
            
        self.distance_a = self.get_distance_array(self.get_sensor_area_list(2)) # thumb  # defined constant as in the base mask
        self.distance_b = self.get_distance_array(self.get_sensor_area_list(3)) # inner_ball
        self.distance_c = self.get_distance_array(self.get_sensor_area_list(4)) # outer_ball
        self.distance_d = self.get_distance_array(self.get_sensor_area_list(5)) # heel
        
        dr = self.draw_grid_color(pressure_data)
        # dr.figure.savefig("E:\\SE7ENDI\\Journals\\Shoe pictures (from conference data)\\{}.png" . format(file_name))
        
        dr = 255 - resize(np.array(dr), (dim, dim, 1))
        return dr