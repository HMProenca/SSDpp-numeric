# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:09:11 2019

@author: gathu
"""

import numpy as np
from copy import deepcopy
from gmpy2 import xmpz,popcount
from numba import jit

from src.mdl.length_encoding import delta_score
from src.util._read_dataset import init_bitset_numeric
#from src.util.fastmean_var import compute_RSS, compute_mean
#import src.util.fastmean_var as fast

class subgroup():
    def __init__(self):
        self.pattern = ()
        self.statistic = []
        self.delta_data = 0 
        self.delta_model = 0
        self.score = 0
        self.usage_total = 0

    def update(self, model,pattern, statistic, score, gain_data,
               gain_model, usage_total):
        self.pattern = pattern
        self.variable_list = [item[0] for item in pattern]
        self.statistic = statistic
        self.gain_data = gain_data
        self.gain_model = gain_model
        self.score = score
        self.usage_total = usage_total
        return self

    def bitset(self,tid_bitsets):
        if self.pattern:
            tid_pattern = tid_bitsets[self.pattern[0]]
            for item in self.pattern:
                tid_pattern &= tid_bitsets[item]
                self.bitset = tid_pattern
            self.support_total = popcount(self.bitset)
        return self

class beam():
    def __init__(self,beam_width, iterative = False):
        self.iterative = iterative
        if self.iterative:
            beam_width= beam_width+1 # because the sugroup2add needs to be removed
            self.model_list = [None for w in range(beam_width)]
            self.forbidden_list = []
        self.beam_width = beam_width
        self.patterns =[() for w in range(beam_width)]
        self.array_score = np.full(beam_width, np.NINF)
        self.min_score = np.NINF
        self.min_index = 0

        
    def replace(self,new_pattern,new_score,model = None):
        self.patterns[self.min_index] = new_pattern
        self.array_score[self.min_index] = new_score
        self.min_index = self.array_score.argmin()
        self.min_score = self.array_score[self.min_index]
        if self.iterative:
            self.model_list[self.min_index] = deepcopy(model)
        
    def clean(self):
        self.patterns =[() for w in range(self.beam_width)]
        self.array_score = np.array([np.NINF for w in range(self.beam_width)])
        self.min_score = np.NINF
        self.min_index = 0
        
    def clean1pattern(self,subgroup2remove):
        if subgroup2remove.pattern:
            idx = self.patterns.index(subgroup2remove.pattern)
            self.patterns[idx] = []
            self.array_score[idx] = 0
            # TODO: not sure if putting self.min_score makes sense
            self.min_score = 0
            self.min_index = idx
            self.forbidden_list.append(subgroup2remove.pattern)
        return self
        
    def return_newmodel(self,tid_bitsets,attributes,original_model):
        # find maximum value
        max_idx = self.array_score.argmax()
        model = self.model_list.pop(max_idx)
        pattern = self.patterns.pop(max_idx)
        self.array_score = np.delete(self.array_score, max_idx)
        self.beam_width = np.count_nonzero(self.array_score)
        # find minimum again
        if model:
            self.min_index = self.array_score.argmin()
            self.min_score = self.array_score[self.min_index]        
            subgroup2expand = subgroup()
            statistic, usage_total = compute_statistic[model.target_type](model,
                                              pattern,tid_bitsets)
            score,gain_data,gain_model = delta_score(model,pattern,
                                                     statistic,usage_total)
            subgroup2expand.update(model,pattern,statistic,score, gain_data,
                                   gain_model, usage_total)
            subgroup2expand.bitset(tid_bitsets)
            model = model.add_rule(subgroup2expand,tid_bitsets,attributes)
        else:
            model = deepcopy(original_model)
        return self, model
        
def find_best_rule(model, data, attributes, tid_bitsets):
    """ find the best rule to add each point using beam search of
    
    """
    #update_tid_bitsets(model,data, attributes,tid_bitsets)
    beam_subgroups,subgroup2add = find_best_singletons(
            model,tid_bitsets)
    
    for depth in range(1,model.max_depth):
        candidates2refine = [sg for sg in beam_subgroups.patterns if sg]
        beam_subgroups.clean()
        for cand in candidates2refine:
            forbidden_list = [item[0] for item in cand]
            tid_cand = rulebitset(cand,tid_bitsets)
            refine_naive(model,cand,tid_cand,forbidden_list,tid_bitsets,
                         beam_subgroups,subgroup2add)
    
    subgroup2add.bitset(tid_bitsets)
    return subgroup2add

def update_tid_bitsets(model,data, attributes,tid_bitsets):
    index_not_consider =  xmpz(model.bitset_covered)
    index_not_consider = list(index_not_consider.iter_set())
    for i_at in attributes:
        if attributes[i_at]["type"] == "numeric":
            # delete cut points from tidbitset and attributes
            for ncut in range(1,attributes[i_at]["ncutpoints"]+1):
                del attributes[i_at][(i_at,ncut)]
                del attributes[i_at][(i_at,-ncut)]
                del tid_bitsets[(i_at,ncut)]
                del tid_bitsets[(i_at,-ncut)]
            init_bitset_numeric(data,attributes,i_at,tid_bitsets,*index_not_consider)
    
def statistic_nominal(model,pattern,tid_bitsets,*other_tid):
    """ args only expects one extra argument which is the bitset
    of the previous pattern to grow (cand), i.e., the pattern we are testing is
    cand + item
    """
    statistic = dict()
    usage = []
    for c in model.class_codes:    
        tid = model.bitset_class[c]
        for item in pattern:
            tid = tid & tid_bitsets[item]
        if other_tid:
            tid = tid & other_tid[0]    
        usage.append(popcount(tid))
    statistic["usage_rule"] =usage 
    usage_total = sum(usage) 
    return statistic, usage_total

def statistic_numeric(model,pattern,tid_bitsets,*other_tid):
    """ args only expects one extra argument which is the bitset
    of the previous pattern to grow (cand), i.e., the pattern we are testing is
    cand + item
    the statistics that define a numeric target are:
        - mean 
        - variance
        - usage
    """
    tid = model.bitset_uncovered 
    for item in pattern:
        tid = tid & tid_bitsets[item]
    if other_tid:
        tid = tid & other_tid[0] 
    statistic = compute_statistic_numeric(model,tid)
    return statistic, statistic["usage"]

def compute_statistic_numeric(model,tid):
    statistic = dict()
    # pattern related part
    aux_bitset = xmpz(tid)
    idx_bits = list(aux_bitset.iter_set())
    values = model.target_values[idx_bits]
    statistic["usage"] = values.size
    if statistic["usage"] > 1:
        #statistic["mean"],closest2,diff2 = compute_mean_and_twopoints(values,model.default_statistic["mean"])
        statistic["mean"] = compute_mean(values)
        closest2,diff2 = find2points(values,model.default_statistic["mean"])
        statistic["mean2"] = compute_mean(closest2)
        statistic["variance2"] = compute_RSS(closest2,statistic["mean2"])/2
        statistic["RSS2"] = compute_RSS(closest2,model.default_statistic["mean"])
        statistic["variance"] = compute_RSS(values,statistic["mean"])/statistic["usage"]
        statistic["RSS_default_pattern"] = compute_RSS(values,model.default_statistic["mean"])
    else:
        statistic["mean"] = 0
        statistic["variance"] = 0
        statistic["variance2"] = 0
        statistic["RSS2"] = 0
        statistic["RSS_default_pattern"] = 0      

    # last rule related part
    bitset_default =  xmpz(model.bitset_uncovered &~ tid)
    idx_bitsdef = list(bitset_default.iter_set())
    values_uncovered = model.target_values[idx_bitsdef]
    statistic["usage_default"] = len(values_uncovered)
    if model.task == "discovery":
        if statistic["usage_default"]:
            statistic["RSS_default_uncovered"] =compute_RSS(values_uncovered,model.default_statistic["mean"])
        else:
            statistic["RSS_default_uncovered"] = 0 
    elif model.task == "classification":    
        statistic["mean_default"] = np.mean(values_uncovered)
        statistic["variance_default"] = np.var(values_uncovered)
    return statistic

@jit(nopython=True)
def compute_RSS(values,meanval):
    c = values-meanval
    RSS =np.dot(c,c)
    return RSS

@jit(nopython=True)
def find2points(values,meandata):
    closest = values[0:2]
    dif = [abs(val-meandata) for val in closest]
    for x in values:
        if abs(x-meandata) < dif[0] and x != closest[1]:
           closest[0] = x
           dif[0] = abs(x-meandata)
        if abs(x-meandata) < dif[1] and x != closest[0]:
           closest[1] = x
           dif[1] = abs(x-meandata)    
    return closest,dif
 
@jit(nopython=True)
def compute_mean(values):
    return np.mean(values)

@jit(nopython=True)
def compute_mean_and_twopoints(values,meandata):
    meanval = 0
    closest = values[0:2]
    dif = [abs(val-meandata) for val in closest]
    for x in values:
        meanval += x
        if abs(x-meandata) < dif[0] and x != closest[1]:
           closest[0] = x
           dif[0] = abs(x-meandata)
        if abs(x-meandata) < dif[1] and x != closest[0]:
           closest[1] = x
           dif[1] = abs(x-meandata)
    meanval = meanval/len(values)
    return meanval,closest,dif

compute_statistic={
	'nominal':statistic_nominal,
	'numeric':statistic_numeric,
};

def find_best_singletons(model,tid_bitsets):
    subgroup2add = subgroup()
    beam_subgroups = beam(model.beam_width)
    for item in tid_bitsets:
        pattern = [item]
        statistic, usage_total = compute_statistic[model.target_type](model,pattern,tid_bitsets)
        score,gain_data,gain_model = delta_score(model,pattern,statistic,usage_total)
        if score > subgroup2add.score:
            subgroup2add.update(model,pattern,statistic,score,gain_data,
               gain_model, usage_total)
        if score > beam_subgroups.min_score:
            beam_subgroups.replace([item],score)
    return beam_subgroups,subgroup2add

def refine_naive(model,cand,tid_cand,forbidden_list,tid_bitsets,beam_subgroups,subgroup2add):
    for item in tid_bitsets:
        if item[0] in forbidden_list: continue
        statistic, usage_total = compute_statistic[model.target_type](model,[item],tid_bitsets,tid_cand)
        newcand = cand +[item]
        score,gain_data,gain_model = delta_score(model,newcand,statistic,usage_total)
        if score > subgroup2add.score:
            subgroup2add.update(model,newcand,statistic,score,gain_data,
               gain_model, usage_total) 
        if score > beam_subgroups.min_score:
            beam_subgroups.replace(newcand,score)    

def rulebitset(pattern,tid_bitsets):
    # THIS CAN BE IMPROVED
    tid_cand = tid_bitsets[pattern[0]]
    for item in pattern:
        tid_cand &= tid_bitsets[item]
    return tid_cand

def rulebitset_support(model,pattern,tid_bitsets): 
    # THIS CAN BE IMPROVED
    tid_cand = tid_bitsets[pattern[0]]
    for item in pattern:
        tid_cand &= tid_bitsets[item]
    return tid_cand

  