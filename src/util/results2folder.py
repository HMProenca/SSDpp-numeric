# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:44:34 2019

@author: gathu
"""

from datetime import datetime
import os
import shutil

def makefolder_time():
    today = datetime.now()
    today.strftime('%Y%m%d_%H%M%S_results')
    folder_path = os.path.join("results",today.strftime('%Y%m%d_%H%M%S_results'))
    os.mkdir(folder_path)
    return folder_path

def makefolder_name(foldername):
    folder_path = os.path.join("results",foldername) 
    if not os.path.exists(folder_path):
    	os.mkdir(folder_path)
    else:
    	if False:
    		shutil.rmtree(folder_path)
    		os.mkdir(folder_path)
    return folder_path

def attach_results(model,string,datasetname):
    string += datasetname + ","
    for meas in model.measures:
        string +=  str(round(model.measures[meas],4)) + ","
    string += " \n"
    return string

def print2folder(model,string,foldername = "time"):
    toprow = "datasetname" + ","
    for meas in model.measures:
        toprow +=  meas + ","
    toprow += " \n"
    
    toprint = toprow+string
      
    if foldername == "time":
        folder_path = makefolder_time()
    elif isinstance(foldername, str):
        folder_path = makefolder_name(foldername)  
    else:
        print("Invalid foldername")
    resultsfile = os.path.join(folder_path,"summary.csv")
    with open(resultsfile, 'w') as file:
        file.write("%s," % toprint)

