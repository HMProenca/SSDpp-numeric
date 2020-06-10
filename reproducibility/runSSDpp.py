# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split

from src.util.results2folder import attach_results,print2folder
from _classes import SSDC

# user configuration
delim = ','
disc_type = "static"
task_name = "discovery"
max_len = 5
beamsize = 100
ncutpoints = 5
target_type = "numeric"
results = ""
datasetnames = ["baseball","autoMPG8","dee","ele-1","forestFires","concrete",\
                "treasury","wizmir","abalone","puma32h","ailerons","elevators",\
                "bikesharing","california","house"]

for datasetname in datasetnames:
    print("Dataset: " + datasetname)
    filename =  "./data/numeric target/"+datasetname+".csv"
    df = pd.read_csv(filename,delimiter=delim)
    model = SSDC(target_type,max_depth=max_len, beam_width = beamsize,
             iterative_beam_width = 1,n_cutpoints = ncutpoints,
             task = task_name,discretization = disc_type,gain="normalized")
    model.fit(df)
    # fit to new data 
    model.measures["nsamples_train"] = len(df)
    results = attach_results(model,results,datasetname)
print2folder(model,results,"SSDpp")
