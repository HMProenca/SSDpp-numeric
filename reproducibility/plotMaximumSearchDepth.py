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
# maximum depth of search
###############################################################################

folder_load = os.path.join("results","maxdepth_results","summary.csv")
folder_save = "maxdepth_plot"
folder_path = makefolder_name(folder_save)
df = pd.read_csv(folder_load,index_col=False)

datasetsnames= np.unique(df.datasetname)
results2plot = dict()
for datname in datasetsnames:
    results2plot[datname] = dict()
    results2plot[datname]["maxdepth"] = df[df.datasetname == datname].maxdepth.to_numpy()
    results2plot[datname]["compression"] = df[df.datasetname == datname].length_ratio.to_numpy()
    results2plot[datname]["time"] = df[df.datasetname == datname].runtime.to_numpy()
    results2plot[datname]["conditions"] = df[df.datasetname == datname].avg_items.to_numpy()

fig,lgd = make_graph(results2plot,"maxdepth","compression",size_marker = 6,color = tableau20,typeofplot ="plot",separate_colour =1)
plt.axvline(x=5,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("maximum depth")
plt.ylabel("relative compression")
save_path = os.path.join(folder_path,"maxdepth_vs_compression.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')

fig,lgd = make_graph(results2plot,"maxdepth","time",size_marker = 6,color = tableau20,typeofplot ="semilogy",separate_colour =1)
plt.axvline(x=5,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("maximum depth")
plt.ylabel("rumtime (s)")
save_path = os.path.join(folder_path,"maxdepth_vs_time.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')

fig,lgd = make_graph(results2plot,"maxdepth","conditions",size_marker = 6,color = tableau20,typeofplot ="plot",separate_colour =1)
plt.axvline(x=5,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("maximum depth")
plt.ylabel("average conditions per subgroup")
save_path = os.path.join(folder_path,"maxdepth_vs_conditions.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')