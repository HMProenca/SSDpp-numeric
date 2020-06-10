# -*- coding: utf-8 -*-

import pandas as pd

from src.util.results2folder import attach_results,print2folder
from _classes import SSDC

###############################################################################
# ncutpoints size experiment
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
ncutpoints_list = [3,5,8,10]

datasetnames = ["baseball","autoMPG8","dee","ele-1","forestFires","concrete",\
                "treasury","wizmir","abalone","puma32h","ailerons","elevators",\
                "bikesharing","california","house"]
results = ""
for datasetname in datasetnames:
    print("Dataset name: " + datasetname)
    for ncutpoints in ncutpoints_list:
        print("   number cut points: " + str(ncutpoints))
        filename =  "./data/numeric target/"+datasetname+".csv"
        df = pd.read_csv(filename,delimiter=delim)
        model = SSDC(target_type,max_depth=max_len, beam_width = beamsize,
                 iterative_beam_width = iterative,n_cutpoints = ncutpoints,
                 task = task_name,discretization = disc_type, gain="normalized")
        model.fit(df)
        model.measures["ncutpoints"]=ncutpoints
        results = attach_results(model,results,datasetname)
results = results.rstrip(", \n")
print2folder(model,results,"ncutpoints_results")

