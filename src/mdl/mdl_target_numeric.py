# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:09:04 2020

@author: gathu
"""

from src.util.general_functions import log2_0
from numpy import NINF,inf

from math import pi




def length_data_numeric(model):
    """ Computes the length of the data L(D|M)
    """    
    ld = 0
    if model.task == "discovery":
        ld += defaultrule_data_length(model)
    elif model.task == "classification":
        #variance = compute variance
        ld += individual_data_length(model,model.default_statistic)
    else: 
        print("Wrong TASK selected")
    for r in range(model.number_rules):
        ld += individual_data_length(model,model.statistic_rules[r])
        #individual_data_lengthprint(model,model.statistic_rules[r])
    return ld

#def individual_data_length(model,statistic):
#    n = statistic["usage"]
#    l_d_ind = 1+n/2*log2_0(pi)-model.l_gamma[n]+0.5*log2_0(n+1)+\
#              n/2*log2_0(n*statistic["variance"])
#    return l_d_ind
def individual_data_lengthprint(model,statistic):
    # refined bayesian encoding of the data
    l_d_opt = refined_bayesian_length(model,statistic["usage"],statistic["variance"])
    # refined Bayesian encoding of first 2 observations (conditional prior)
    l_d_2 = -refined_bayesian_length(model,2,statistic["variance2"])
    # non-refined encoding of first two observations
    l_d_2 += model.l_mean
    l_d_2 += 0.5*model.l_e*statistic["RSS2"]/model.default_statistic["variance"] 
    print("Optimal: " + str(l_d_opt))
    print("cost: " + str(l_d_2))
        


def individual_data_length(model,statistic):
    if statistic["variance"] == 0:
        l_d_ind = inf
    else:        
        # refined bayesian encoding of the data
        l_d_ind = refined_bayesian_length(model,statistic["usage"],statistic["variance"])
        # refined Bayesian encoding of first 2 observations (conditional prior)
        l_d_ind += -refined_bayesian_length(model,2,statistic["variance2"])
        # non-refined encoding of first two observations
        l_d_ind += model.l_mean
        l_d_ind += 0.5*model.l_e*statistic["RSS2"]/model.default_statistic["variance"]     
    return l_d_ind

def refined_bayesian_length(model,n,variance):
    l = 1+n/2*log2_0(pi)-model.l_gamma[n]+0.5*log2_0(n+1)+\
              n/2*log2_0(n*variance)
    return l


def defaultrule_data_length(model):
    nr = model.number_rules
    #print("number of rules " + str(nr))
    #print("statistics rules " + str(len(model.statistic_rules)))
    if model.task == "classification":
        l_d_default =individual_data_length(model,model.default_statistic)
    elif model.task == "discovery":
        if nr !=0: 
            l_d_default = 0.5*model.support_uncovered*model.l_mean
            l_d_default +=0.5*model.l_e*model.statistic_rules[nr-1]["RSS_default_uncovered"]/model.default_statistic["variance"]
        else: 
            l_d_default = model.default_statistic["usage"]/2*(model.l_mean+model.l_e)
    return l_d_default

def delta_data_numeric_independent(model,statistic):
    """ Used assuming that the rules are independent, 
    Computes the delta code length of the data \Delta L(D|M) using NML code
    of adding one rule to the model
    """
    if model.task == "classification":
        dld = model.constant -individual_data_length(model, statistic)
        statistic_newdefault ={"variance":statistic["variance_default"],
                               "usage": statistic["usage_default"]}
        dld += -individual_data_length(model, statistic_newdefault)
    elif model.task == "discovery":
        l_default_usage = 0.5*statistic["usage_default"]*model.l_mean+\
            0.5*model.l_e*statistic["sse_default_support"]/model.default_statistic["variance"]
        l_default = 0.5*statistic["usage"]*model.l_mean+\
            0.5*model.l_e*statistic["sse_default_support"]/model.default_statistic["variance"]        
        #l_default = 0.5*statistic["usage"]*model.l_mean+\
        # 0.5*model.l_e*statistic["sse_default"]/model.default_statistic["variance"] # sequential 
        if l_default_usage < 0:
            dld = NINF           
        else:    
            dld = l_default          
    else:
        print("Wrong task selected") 
    # new pattern 
    if statistic["usage"]:
        l_individual = individual_data_length(model, statistic)
        #print("rule length: " + str(l_individual) + "| default length: " + str(l_default))
        dld += -l_individual
    return dld

def delta_data_numeric(model,statistic):
    """ Computes the delta code length of the data \Delta L(D|M) using NML code
    of adding one rule to the model
    """    
    if model.task == "classification":
        dld = model.constant -individual_data_length(model, statistic)
        statistic_newdefault ={"variance":statistic["variance_default"],
                               "usage": statistic["usage_default"]}
        dld += -individual_data_length(model, statistic_newdefault)
    elif model.task == "discovery":
        l_default = 0.5*statistic["usage"]*model.l_mean
        l_default +=0.5*model.l_e*statistic["RSS_default_pattern"]/model.default_statistic["variance"] # sequential 
        dld = l_default        
    else:
        print("Wrong task selected") 
    if statistic["usage"]:
        # new pattern 
        l_individual = individual_data_length(model, statistic)
        dld += -l_individual 
    return dld

def delta_data_const_numeric(model):
    """ Computes the constant related with L(D|M) and \Delta L(D|M) which is 
    equal to the length of the data covered by the default rule and encoded 
    by this - L(D default rule|default rule). This is a constant in the greedy
    adding of one rule \Delta L(D|M)
    """      
    if model.task == "discovery":
        const = defaultrule_data_length(model)
    elif model.task == "classification":
        const= individual_data_length(model,model.default_statistic)
    else:
        print("WRONG CHOICE OF TASK")    
    return const