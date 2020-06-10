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
# beamsize 
###############################################################################

folder_load = os.path.join("results","beamsize_results","summary.csv")
folder_save = "beamsize_plot"
folder_path = makefolder_name(folder_save)
df_beam = pd.read_csv(folder_load,index_col=False)

datasetsnames= np.unique(df_beam.datasetname)
results2plot = dict()
for datname in datasetsnames:
    results2plot[datname] = dict()
    results2plot[datname]["beamsize"] = df_beam[df_beam.datasetname == datname].beamsize.to_numpy()
    results2plot[datname]["compression"] = df_beam[df_beam.datasetname == datname].length_ratio.to_numpy()
    results2plot[datname]["time"] = df_beam[df_beam.datasetname == datname].runtime.to_numpy()

fig,lgd = make_graph(results2plot,"beamsize","compression",size_marker = 6,color = tableau20,typeofplot ="semilogx",separate_colour =1)
plt.axvline(x=100,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("beam width")
plt.ylabel("relative compression")
save_path = os.path.join(folder_path,"beam_vs_compression.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')

fig,lgd = make_graph(results2plot,"beamsize","time",size_marker = 6,color = tableau20,typeofplot ="loglog",separate_colour =1)
plt.axvline(x=100,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("beam width")
plt.ylabel("rumtime (s)")
save_path = os.path.join(folder_path,"beam_vs_time.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')