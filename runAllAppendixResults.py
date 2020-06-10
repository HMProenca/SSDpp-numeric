# -*- coding: utf-8 -*-
###############################################################################
#
#
#                           APPENDIX RESULTS 
#
#
###############################################################################
"""
ABSOLUTE VS NORMALIZED gain experiments
runExperimentGain.py - it runs the experiments for all datasets with normalized
and absolute gain. It prints all results to 2 folders in the results file:
    - /results/normalized_results/summary.csv
    - /results/absolute_results/summary.csv

plotExperimentGain.py - reads the results in previous files and makes 4 plots, 
for compression, runtime, Sum of Weighted Kullback-Leibler, number of rules, to
the following files:
    - /results/normalizedvsabsolute/normalizedvsabsolute_compression.pdf
    - /results/normalizedvsabsolute/normalizedvsabsolute_runtime.pdf
    - /results/normalizedvsabsolute/normalizedvsabsolute_SWKL.pdf
    - /results/normalizedvsabsolute/normalizedvsabsolute_rules.pdf   
"""
exec(open("./reproducibility/runExperimentGain.py").read())
exec(open("./reproducibility/plotExperimentGain.py").read())

"""
BEAM WIDTH SIZE experiments
    -"./reproducibility/runBeamSize.py" - it runs the experiments for different
        values of beam size for all datasets and prints the results to:
        "/results/beamsize_results/summary.csv
    - "./reproducibility/plotBeamSize.py" - reads the previous summary.csv
        and plots the results regarding the compression, runtime to:
        /results/beamsize_plot/beam_vs_compression.pdf
        /results/beamsize_plot/beam_vs_time.pdf
"""
# Beam size analysis experiments
exec(open("./reproducibility/runBeamSize.py").read())
exec(open("./reproducibility/plotBeamSize.py").read())

"""
NUMBER OF CUTPOINTS FOR NUMERIC VARIABLES experiments
    -"./reproducibility/runNumberCutPoints.py" - it runs the experiments 
    for different values of beam size for all datasets and prints 
    the results to:
        "/results/beamsize_results/summary.csv
    - "./reproducibility/plotNumberCutPoints.py" - reads the previous 
    summary.csv and plots the results regarding the compression, runtime to:
        /results/ncutpoints_plot/ncutpoints_vs_compression.pdf
        /results/ncutpoints_plot/ncutpoints_vs_time.pdf
"""
# Number of cut points analysis experiments
exec(open("./reproducibility/runNumberCutPoints.py").read())
exec(open("./reproducibility/plotNumberCutPoints.py").read())

"""
MAXIMUM DEPTH OF BEAM SEARCH experiments
    -"./reproducibility/runNumberCutPoints.py" - it runs the experiments 
    for different values of beam size for all datasets and prints 
    the results to:
        "/results/beamsize_results/summary.csv
    - "./reproducibility/plotNumberCutPoints.py" - reads the previous 
    summary.csv and plots the results regarding the compression, runtime to:
        /results/ncutpoints_plot/ncutpoints_vs_compression.pdf
        /results/ncutpoints_plot/ncutpoints_vs_time.pdf
"""
# Run maximum search depth analysis experiments
exec(open("./reproducibility/runMaximumSearchDepth.py").read())
exec(open("./reproducibility/plotMaximumSearchDepth.py").read())

