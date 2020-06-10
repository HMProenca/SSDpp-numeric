# -*- coding: utf-8 -*-

from scipy.special  import comb
import numpy as np
from math import log,log2,exp,pi,sqrt
from gmpy2 import xmpz,mpz,popcount
from scipy.special import gammaln
from operator import itemgetter

from src.mdl.length_encoding import (universal_code_integers,
                                     multinomial_with_recurrence,
                                     compute_length_data,compute_length_model,
                                     delta_data_const)
from src.measures.subgroupdiscovery import (jaccard_index_model,
                                            nominal_discovery_measures,
                                            numeric_discovery_measures)

from src.beam.itemset_beamsearch import find_best_rule, beam


#########################################################

class rulelist_numeric():
    """ rule set model

    """
    def __init__(self, data,attributes,target,types,task,max_depth,
                 beam_width,iterative_beam_width,max_rules,gain):
        self.task = task
        self.number_rules = 0
        self.beam_width = beam_width
        self.gain = gain
        self.max_depth = max_depth
        self.iterative_beam_width = iterative_beam_width
        self.max_rules = max_rules
        self.default_statistic = {"mean": target["mean"],
                                   "variance": target["variance"],
                                   "usage": target["n_rows"]}
        self.create_constants(data,attributes,target,types)
        self.target_type = target['type'] 
        self.target_values = target["values"]
        self.number_instances = len(target["values"])
        self.absolute = 0
        # instances covered by rules and not covered
        self.bitset_covered = mpz()
        self.support_covered = 0
        self.bitset_uncovered = target["bitset"] 
        self.support_uncovered = target["n_rows"]
        self.pattern4prediction = []
        self.antecedent_raw = []
        self.antecedent_description = []
        self.consequent_description = []
        self.statistic_rules = []
        self.support_rules = []
        self.bitset_rules = []
        self.rule_scores = []
        self.length_model = 0
        self.length_data = compute_length_data[self.target_type](self)
        self.length_original =  self.length_data      
        self.constant = delta_data_const[self.target_type](self)
    
    def add_rule(self,subgroup2add,tid_bitsets,attributes):
        
        self.number_rules += 1
        tid_cand = subgroup2add.bitset
        self.bitset_covered = self.bitset_covered | tid_cand 
        self.support_covered = popcount(self.bitset_covered)
        self.bitset_uncovered = self.bitset_uncovered &~ tid_cand
        self.support_uncovered = popcount(self.bitset_uncovered)
        self.bitset_rules.append(tid_cand)
        self.antecedent_raw.append(subgroup2add.pattern)        
        self.statistic_rules.append(subgroup2add.statistic)
        # IN CLASSIFICATION CASE EVERYTHING HAS TO BE UPDATED!
        self.default_statistic["usage"] = self.support_uncovered
        if self.task == "classification":
            aux_tid = xmpz(self.bitset_uncovered)
            idx_bitsdef = list(aux_tid.iter_set())
            values_default = self.target_values[idx_bitsdef]     
            self.default_mean["mean"] = np.mean(values_default)
            self.default_variance["variance"] = np.var(values_default)
        # SHOULD BE REMOVED LATER ON
        support = popcount(tid_cand)
        self.support_rules.append(support)
        self.add_pattern4prediction(subgroup2add.pattern,attributes)
        self.consequent_description.append(self.add_description_consequent(subgroup2add))
        self.consequent_lastrule_description = self.add_consequent_lastrule()
        self.add_description_antecedent(subgroup2add.pattern,attributes)  
        self.length_model = compute_length_model(self)
        self.length_data = compute_length_data[self.target_type](self)
        self.constant = delta_data_const[self.target_type](self)
        if self.length_original > 0:
            self.length_ratio = (self.length_data+self.length_model)/self.length_original
        elif self.length_original < 0:
            self.length_ratio = self.length_original/(self.length_data+self.length_model)
        return self
       
    def create_constants(self,data,attributes,target,types):
        # TODO : take this arbitrary 500 out of here!
        self.l_universal = {key: universal_code_integers(key) for key in range(0,500)}
        #compute the gamma for n points which is f(n) = log Gamma((n-1)/2)
        le2 =log(2)
        # constant that is used for the calculations of data encoding
        self.l_mean = log2(2*pi*self.default_statistic["variance"])
        self.l_e = log2(exp(1))
        self.l_gamma = {key: gammaln(key/2)/le2 if key > 0 else 0 
                        for key in range(0,data.shape[0]+1)}                 
        n_variables = sum([len(vars) for vars in types.values()])
        self.l_comb = {key: log2(comb(n_variables,key))
                  for key in range(1,self.max_depth+1)}
        self.l_var = {iat: log2(len(attributes[iat]["label_code"])) for iat in attributes}
        return self
    
    def add_description_antecedent(self,pattern,attributes):
        text2add = ""
        for idx,item in enumerate(pattern):
            att = item[0]
            type = attributes[att]["type"]
            variable_name = attributes[att]["attribute_name"]
            if type == "numeric":
                if attributes[att][item][0] == "minvalue":
                    text2add += variable_name +" >= "+str(attributes[att][item][1])
                elif  attributes[att][item][0] == "maxvalue": 
                    text2add += variable_name +" <= "+str(attributes[att][item][1])
                elif  attributes[att][item][0] == "interval": 
                    minval = attributes[att][item][1][0]
                    maxval = attributes[att][item][1][1]
                    text2add +=str(minval)+" <= "+variable_name+" <= "+str(maxval)                
                else:
                    print("Wrong terms for antecedent description")
            else:
                text2add += variable_name + " = " + attributes[att][item]  
            if idx == len(pattern)-1:
                pass
            else:
                text2add += " AND "
        self.antecedent_description.append(text2add)
        return self

    def add_pattern4prediction(self,pattern,attributes):
        subsetdefinition = {"type":[],"var_name": [], 
                            "variable": [],"subset":[], 
                            "column":  [],"nitems" : 0}
        for idx,item in enumerate(pattern):
            att = item[0]
            type = attributes[att]["type"]
            subset = attributes[att][item]
            var_name = attributes[att]["attribute_name"]
            subsetdefinition["type"].append(type)
            subsetdefinition["var_name"].append(var_name)
            subsetdefinition["subset"].append(subset)
            subsetdefinition["column"].append(item[0])
            subsetdefinition["nitems"] += 1
        self.pattern4prediction.append(subsetdefinition)
        return self
    
    def add_description_consequent(self,subgroup):
        text2add = "mean = " + str(subgroup.statistic["mean"]) +\
                    "; std = " + str(sqrt(subgroup.statistic["variance"])) +\
                    " , "
                    #"; variance = " + str(subgroup.statistic["variance"]) +\
                    #"; usage = " + str(subgroup.statistic["usage"]) +\
                    #"; usage default = " + str(subgroup.statistic["usage_default"]) +\
                    #"; RSS subgroup = " + str((subgroup.statistic["usage"]-1)*subgroup.statistic["variance"]) +\
                    #"; RSS default = " + str(subgroup.statistic["RSS_default_pattern"]) +\
        return text2add
    
    def add_consequent_lastrule(self):
        if self.task == "discovery":
            text2add = "mean = " + str(self.default_statistic["mean"]) +\
                       "std  = " + str(sqrt(self.default_statistic["variance"]))
                       #"variance  = " + str(self.default_statistic["variance"]) +\
                       #"usage = " + str(self.default_statistic["usage"])
        elif self.task == "classification":
            pass
        return text2add        
            
    def check_constraints(self):
        comply_contraint = True
        return comply_contraint
                
    
    def compute_jaccard_index(self,tid_bitsets):
        self.jaccard_matrix = jaccard_index_model(self,tid_bitsets)
    # remove empty rule column and row
    def subgroup_discovery_measures(self):
        self.measures = numeric_discovery_measures(self)

#def find_rulelist(data,attributes,target,types,tid_bitsets,task,max_depth,beam_width,iterative_beam_width,constraints):
#    # initialize model
#    if target["type"] == "nominal": 
#        model = rulelist_nominal(data,attributes,target,types,task,max_depth,beam_width,iterative_beam_width)
#    elif target["type"] == "numeric":
#        model = rulelist_numeric(data,attributes,target,types,task,max_depth,beam_width,iterative_beam_width,constraints)
#    model_list = []
#    iterative_subgroups = beam(model.iterative_beam_width, iterative = True)
#    iter = 1
#    print("Iteration: " + str(iter))
#    # Normalized model iteration, first estimation
#    while True:
#        subgroup2add = find_best_rule(model, data, attributes, tid_bitsets,iterative_subgroups) 
#        comply_constraint = model.check_constraints("number_rules",model.number_rules)
#        #if subgroup2add.score <= 0 or not comply_constraint: break
#        if subgroup2add.score <= 0: break
#        #print("score: " + str(subgroup2add.score))
#        print(subgroup2add.statistic)
#        model = model.add_rule(subgroup2add,tid_bitsets,attributes)
#    model.compute_jaccard_index(tid_bitsets)
#    model.subgroup_discovery_measures()
#    model_list.append(model)
#    
#    old_rule_bitsets = []
#    tiduncovered = target["bitset"]
#    for tid in model.bitset_rules:
#        old_rule_bitsets.append(tid & tiduncovered)       
#        tiduncovered = tiduncovered &~ tid
#    
#    #old_rule_bitsets.append(model.bitset_uncovered)
#    oldscores = model.rule_scores
#    #oldscores= [(-score*model.statistic_rules[idx]["usage"]+model.statistic_rules[idx]["RSS_default_pattern"])/model.statistic_rules[idx]["usage"] for idx,score in enumerate(model.rule_scores)]
#    #oldscores.append(model.statistic_rules[-1]["RSS_default_uncovered"]/model.statistic_rules[-1]["usage_default"])
#    # initialize model
#    if target["type"] == "nominal": 
#        model = rulelist_nominal(data,attributes,target,types,task,max_depth,beam_width,iterative_beam_width)
#    elif target["type"] == "numeric":
#        model = rulelist_numeric(data,attributes,target,types,task,max_depth,beam_width,iterative_beam_width,constraints)
#    # iterative with an estimation
#    model.oldtid = old_rule_bitsets 
#    model.estimation=oldscores
#    while True:
#        subgroup2add = find_best_rule(model, data, attributes, tid_bitsets,iterative_subgroups)
#        print("theses one")
#        print(subgroup2add.statistic)
#        comply_constraint = model.check_constraints("number_rules",model.number_rules)
#        if subgroup2add.gain_data + subgroup2add.gain_model<= 0: break
#        #print("score: " + str(subgroup2add.score))
#        print(subgroup2add.statistic)
#        model = model.add_rule(subgroup2add,tid_bitsets,attributes)
#    model.compute_jaccard_index(tid_bitsets)
#    model.subgroup_discovery_measures()
#    model_list.append(model)    
#    score_list = [auxm.length_ratio for auxm in model_list]
#    print(score_list)
#    print(model.number_rules)
#    index, value = min(enumerate(score_list), key=itemgetter(1))
#    from matplotlib.pyplot import plot
#    plot(score_list)
#    model = model_list[index]
#    return model


def find_rulelist(data,attributes,target,types,tid_bitsets,task,max_depth,
                  beam_width,iterative_beam_width,max_rules,gain):
    # initialize model
    model = rulelist_numeric(data,attributes,target,types,task,max_depth,
                             beam_width,iterative_beam_width,
                             max_rules,gain)
    while True:
        print("Iteration: " + str(model.number_rules+1))
        subgroup2add = find_best_rule(model, data, attributes, tid_bitsets) 
        if subgroup2add.score <= 0: break
        model = model.add_rule(subgroup2add,tid_bitsets,attributes)
    model.compute_jaccard_index(tid_bitsets)
    model.subgroup_discovery_measures()
    return model
