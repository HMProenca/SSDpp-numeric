# -*- coding: utf-8 -*-

import pandas as pd
import sys
sys.path
from src.util.results2folder import attach_results,print2folder

from _classes import SSDC

# load data
datasetname= "baseball"
delim = ','
filename =  "./data/numeric target/"+datasetname+".csv"
df = pd.read_csv(filename,delimiter=delim)

# user configuration
task_name = "discovery"
target_type = "numeric"

# load class and fit to data
model = SSDC(task = task_name)
model.fit(df)
#print("model measures : " +str(model.measures) + "\n")
print(model)
