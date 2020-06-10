# -*- coding: utf-8 -*-


import pandas as pd

from src.util.results2folder import attach_results,print2folder
from _classes import SSDC
###############################################################################
# Absolute vs Normalized code
###############################################################################
# user configuration
delim = ','
disc_type = "static"
task_name = "discovery"
max_len = 5
beamsize = 100
ncutpoints = 5
iterative = 1
target_type = "numeric"
datasetnames = ["cholesterol","baseball","autoMPG8","dee","ele-1","forestFires","concrete",\
                "treasury","wizmir","abalone","puma32h","ailerons","elevators",\
                "bikesharing","california","house"]
#run normalized gain
results = ""
print("Type of Gain : NORMALIZED")
for datasetname in datasetnames:
    print("Dataset name: " + datasetname)
    filename =  "./data/numeric target/"+datasetname+".csv"
    df = pd.read_csv(filename,delimiter=delim)
    model = SSDC(target_type,max_depth=max_len, beam_width = beamsize,
             iterative_beam_width = iterative,n_cutpoints = ncutpoints,
             task = task_name,discretization = disc_type, gain="normalized")
    model.fit(df)
    results = attach_results(model,results,datasetname)
results = results.rstrip(", \n")
print2folder(model,results,"normalized_results")

# run absolute gain
results = ""
print("Type of Gain : ABSOLUTE")
for datasetname in datasetnames:
    print("Dataset name: " + datasetname)
    filename =  "./data/numeric target/"+datasetname+".csv"
    df = pd.read_csv(filename,delimiter=delim)
    model = SSDC(target_type,max_depth=max_len, beam_width = beamsize,
             iterative_beam_width = iterative,n_cutpoints = ncutpoints,
             task = task_name,discretization = disc_type, gain="absolute")
    model.fit(df)
    results = attach_results(model,results,datasetname)
results = results.rstrip(", \n")
print2folder(model,results,"absolute_results")