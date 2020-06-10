# -*- coding: utf-8 -*-
###############################################################################
#
#
#                           PAPER experiments 
#
#
###############################################################################
"""
Run all algorithms, namely, topk, seqcover, SSD++ and compare their results.
The following functions run the files for each algorithm, make the 
corresponding results tables and plots.
    - "./reproducibility/runSSDpp.py" - run SSD++ and saves the results
    in "./results/SSDpp/summary.csv" 
    
    - "./reproducibility/runtopk.py" - run DSSD algorithm in the topk mode,
    and saves the results in: "./results/topk/summary.csv" 
    

    - "./reproducibility/runseqcover.py" - run DSSD algorithm in the 
    sequential covering mode, and saves the results in:
    "./results/seqcover/summary.csv" 

"""
# run SSD++ for all datasets
exec(open("./reproducibility/runSSDpp.py").read())
# run seq-cover for all datasets
exec(open("./reproducibility/runDSSDseqcover.py").read())
# run topk for all datasets
exec(open("./reproducibility/runDSSDtopk.py").read())
# mkae plots of runtime and jaccard based on the previous files
exec(open("./reproducibility/plotComparisons.py").read())


# run Hotel application
exec(open("./reproducibility/plotComparisons.py").read())

