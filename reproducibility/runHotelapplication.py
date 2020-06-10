# -*- coding: utf-8 -*-


import pandas as pd
import os
from _classes import SSDC
from src.util.results2folder import makefolder_name
# user configuration
datasetname= "hotel_bookings"
delim = ','
disc_type = "static"
task_name = "discovery"
max_len = 5
beamsize = 100
ncutpoints = 5
target_type = "numeric"
# load data
filename =  "./data/application/"+datasetname+".csv"
df = pd.read_csv(filename,delimiter=delim)

# subset dataset
#df = df.loc[df['hotel'] == "City Hotel",:]
dfaux = df.loc[df['hotel'] == "Resort Hotel",:]

df = df.loc[df['arrival_date_year'] == 2016,:]
#dfaux = dfaux.loc[dfaux['is_canceled'] == 0,:]

def repeated_guest(df):
    """ Test Function for generating new value"""
    if df['is_repeated_guest'] == 0:
        return "no"
    elif df['is_repeated_guest'] == 1:
        return "yes"
     
def months_year(df):
    """ Test Function for generating new value"""
    if df['arrival_date_month'] == "January":
        return 1
    elif df['arrival_date_month'] == "February":
        return 2    
    elif df['arrival_date_month'] == "March":
        return 3     
    elif df['arrival_date_month'] == "April":
        return 4      
    elif df['arrival_date_month'] == "May":
        return 5      
    elif df['arrival_date_month'] == "June":
        return 6      
    elif df['arrival_date_month'] == "July":
        return 7      
    elif df['arrival_date_month'] == "August":
        return 8      
    elif df['arrival_date_month'] == "September":
        return 9      
    elif df['arrival_date_month'] == "October":
        return 10      
    elif df['arrival_date_month'] == "November":
        return 11     
    elif df['arrival_date_month'] == "December":
        return 12  
    
df["is_repeated_guest"] = df.apply(repeated_guest, axis=1)
df["arrival_date_month"] = df.apply(months_year, axis=1)

#dfaux = dfaux.loc[dfaux['is_canceled'] == 1,:]

df["lead_time1"]=df["lead_time"]
df = df.drop(["is_canceled","lead_time",'hotel',"arrival_date_year",
                    "arrival_date_week_number",'reservation_status',
                    "reservation_status_date","booking_changes","adr",
                    "assigned_room_type","agent","company","days_in_waiting_list"], axis=1)
df["lead_time"]=df["lead_time1"]
df = df.drop(["lead_time1"], axis=1)
df.reset_index(inplace = True)
df.to_csv('HotelBookingResort2016.csv', index=False) 


import pandas as pd
from _classes import SSDC

# user configuration
datasetname= "HotelBookingResort2016"
delim = ','
disc_type = "static"
task_name = "discovery"
max_len = 5
beamsize = 100
ncutpoints = 5
target_type = "numeric"
# load data
filename =  "./data/application/"+datasetname+".csv"
df = pd.read_csv(filename,delimiter=delim)
  
# load model
model = SSDC(target_type,max_depth=max_len, beam_width = beamsize,
             iterative_beam_width = 1,n_cutpoints = ncutpoints,
             task = task_name,discretization = disc_type, gain="normalized", max_rules = 4)
model.fit(df)
print(model)

#ruleset = model.rule_sets
#overlap = []
#subgroup_sets_support= [set(rset) for rset in ruleset]
#rules_usg = []
#subgroup_sets_usage= []
#statistic_rules = []
#for r in range(len(subgroup_sets_support)):
#    previous_sets = [subgroup_sets_support[ii] for ii in range(r)]
#    auxset= subgroup_sets_support[r].difference(*previous_sets)
#    print("overlap : " + str(len(subgroup_sets_support[r])-len(auxset)))
#    usage = len(auxset)
#    subgroup_sets_usage.append(auxset)     
#    rules_usg.append(usage)

import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
name_save = "HotelBookingplot"
folder_path = makefolder_name(name_save)
mean_rules = [stat["mean"] for stat in model.statistics]
yvalues = df.iloc[:,-1]
yfirstrule = yvalues[ruleset[1]]
sns.kdeplot(yvalues,label="dataset", shade=True)
#sns.distplot(yvalues,label="dataset distribution")

#plt.axvline(400, 0,0.007)
for irule,meanvalue in enumerate(mean_rules):
    label_name = "$\hat{\mu}_" + str(irule) + "$" 
    plt.plot([meanvalue, meanvalue], [0, 0.006],label=label_name)
plt.ylim(0, 0.009)
plt.xlim(-5, 800)
plt.xlabel("Lead time of booking (days) before arrival")
plt.ylabel("probability density of dataset distribution")
plt.legend();
save_path = os.path.join(folder_path,name_save+ ".pdf")
plt.savefig(save_path, bbox_inches='tight')
