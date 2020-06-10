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
# Absolute vs Normalized plots! compression ratio
###############################################################################
folder_normalized = os.path.join("results","normalized_results","summary.csv")
folder_absolute = os.path.join("results","absolute_results","summary.csv")
name_save = "normalizedvsabsolute_compression"
folder = "plot_normalizedvsabsolute"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder,name_save+".pdf")

df_norm= pd.read_csv(folder_normalized,index_col=False)
df_abs = pd.read_csv(folder_absolute,index_col=False)

labels = df_norm.datasetname.to_numpy()
compression_norm = df_norm.length_ratio.to_numpy()
compression_abs = df_abs.length_ratio.to_numpy()

x = np.arange(len(labels))  # the label locations
width_size = 0.35  # the width of the bars
color1=np.array(tableau20[0])/255
color2=np.array(tableau20[2])/255
fig, ax = plt.subplots()
rects1 = ax.bar(x - width_size/2, compression_norm,
                width= width_size,color = color1, label='normalized')
rects2 = ax.bar(x + width_size/2, compression_abs,width= width_size,
                color = color2, label='absolute')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('compression ratio')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})  
ax.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')

###############################################################################
# Absolute vs Normalized plots! runtime
###############################################################################
folder_normalized = os.path.join("results","normalized_results","summary.csv")
folder_absolute = os.path.join("results","absolute_results","summary.csv")
name_save = "normalizedvsabsolute_runtime"
folder = "plot_normalizedvsabsolute"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder,name_save+".pdf")

df_norm= pd.read_csv(folder_normalized,index_col=False)
df_abs = pd.read_csv(folder_absolute,index_col=False)

labels = df_norm.datasetname.to_numpy()
compression_norm = df_norm.runtime.to_numpy()/60
compression_abs = df_abs.runtime.to_numpy()/60

x = np.arange(len(labels))  # the label locations
width_size = 0.35  # the width of the bars
color1=np.array(tableau20[0])/255
color2=np.array(tableau20[2])/255
fig, ax = plt.subplots()
rects1 = ax.bar(x - width_size/2, compression_norm,
                width= width_size,color = color1, label='normalized')
rects2 = ax.bar(x + width_size/2, compression_abs,width= width_size,
                color = color2, label='absolute')
ax.set_yscale('log')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('runtime (minutes)')
#ax.set_title('dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})  
ax.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')


###############################################################################
# Absolute vs Normalized plots!
###############################################################################
folder_normalized = os.path.join("results","normalized_results","summary.csv")
folder_absolute = os.path.join("results","absolute_results","summary.csv")
name_save = "normalizedvsabsolute_SWKL"
#folder_path = makefolder_name(name_save)
folder = "plot_normalizedvsabsolute"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder,name_save+".pdf")
datainstances = [297,
337,
392,
365,
495,
517,
1030,
1049,
1461,
4177,
8192,
13750,
16599,
17379,
20640,
22784]

df_norm= pd.read_csv(folder_normalized,index_col=False)
df_abs = pd.read_csv(folder_absolute,index_col=False)

labels = df_norm.datasetname.to_numpy()
compression_norm = df_norm.wkl_sum.to_numpy()
compression_norm = [val/datainstances[ival] for ival,val in enumerate(compression_norm)]
compression_abs = df_abs.wkl_sum.to_numpy()
compression_abs = [val/datainstances[ival] for ival,val in enumerate(compression_abs)]

x = np.arange(len(labels))  # the label locations
width_size = 0.35  # the width of the bars
color1=np.array(tableau20[0])/255
color2=np.array(tableau20[2])/255
fig, ax = plt.subplots()
rects1 = ax.bar(x - width_size/2, compression_norm,
                width= width_size,color = color1, label='normalized')
rects2 = ax.bar(x + width_size/2, compression_abs,width= width_size,
                color = color2, label='absolute')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SWKL (normalized per $|D|$')
#ax.set_title('dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})  
ax.legend(loc='best')

fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')
###############################################################################
# absolute vs normalize number of rules 
###############################################################################
folder_normalized = os.path.join("results","normalized_results","summary.csv")
folder_absolute = os.path.join("results","absolute_results","summary.csv")
name_save = "normalizedvsabsolute_rules"
folder = "plot_normalizedvsabsolute"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder,name_save+".pdf")

df_norm= pd.read_csv(folder_normalized,index_col=False)
df_abs = pd.read_csv(folder_absolute,index_col=False)

labels = df_norm.datasetname.to_numpy()
compression_norm = df_norm.n_rules.to_numpy()
compression_abs = df_abs.n_rules.to_numpy()

x = np.arange(len(labels))  # the label locations
width_size = 0.35  # the width of the bars
color1=np.array(tableau20[0])/255
color2=np.array(tableau20[2])/255

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)

rects1_top = ax.bar(x - width_size/2, compression_norm,
                width= width_size,color = color1, label='normalized')
rects1_bottom = ax2.bar(x - width_size/2, compression_norm,
                width= width_size,color = color1, label='normalized')

rects2_top = ax.bar(x + width_size/2, compression_abs,width= width_size,
                color = color2, label='absolute')
rects2_bottom = ax2.bar(x + width_size/2, compression_abs,width= width_size,
                color = color2, label='absolute')
ax.set_ylim(110, 290)  # outliers only
ax2.set_ylim(0, 48)  # most of the data

ax.spines['bottom'].set_visible(False)

ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.grid(False)
fig.text(0.00, 0.6, 'number of rules', va='center', rotation='vertical')

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax2.set_xticks(x)
ax2.set_xticklabels(labels,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})  
ax.legend(loc='best')

#ax.legend()
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')