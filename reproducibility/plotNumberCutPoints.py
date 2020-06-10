# -*- coding: utf-8 -*-

from reproducibility.makegraphs import tableau20,make_graph
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.markers as marker
import matplotlib.axes as axes

from matplotlib.ticker import FormatStrFormatter
from src.util.results2folder import makefolder_name
###############################################################################
# number of cut points 
###############################################################################

folder_load = os.path.join("results","ncutpoints_results","summary.csv")
folder_save = "ncutpoints_plot"
folder_path = makefolder_name(folder_save)
df_ncut = pd.read_csv(folder_load,index_col=False)

datasetsnames= np.unique(df_ncut.datasetname)
results2plot = dict()
for datname in datasetsnames:
    results2plot[datname] = dict()
    results2plot[datname]["ncutpoints"] = df_ncut[df_ncut.datasetname == datname].ncutpoints.to_numpy()
    results2plot[datname]["compression"] = df_ncut[df_ncut.datasetname == datname].length_ratio.to_numpy()
    results2plot[datname]["time"] = df_ncut[df_ncut.datasetname == datname].runtime.to_numpy()

fig,lgd = make_graph(results2plot,"ncutpoints","compression",size_marker = 6,color = tableau20,typeofplot ="plot",separate_colour =1)
plt.axvline(x=5,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("number cutpoints")
plt.ylabel("relative compression")
save_path = os.path.join(folder_path,"ncutpoints_vs_compression.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')

fig,lgd = make_graph(results2plot,"ncutpoints","time",size_marker = 6,color = tableau20,typeofplot ="semilogy",separate_colour =1)
plt.axvline(x=5,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("number cutpoints")
plt.ylabel("rumtime (s)")
save_path = os.path.join(folder_path,"ncutpoints_vs_time.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
