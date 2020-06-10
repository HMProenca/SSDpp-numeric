# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:34:06 2019

@author: gathu
"""

from math import log,ceil, sqrt
import numpy as np

from src.mdl.mdl_target_nominal import (length_data_nominal,
                                        delta_data_nominal,
                                        delta_data_const_nominal)
from src.mdl.mdl_target_numeric import (length_data_numeric,
                                        delta_data_numeric,
                                        delta_data_const_numeric)

   
def multinomial_with_recurrence(L, n):
    """ Computes the Normalized Maximum Likelihood (NML) code length
    L - number of categories for a categorical or multinomial distribution
    n - number of points / samples
    total  - COMP(L,n)
    """
    total = 1.0
    b = 1.0
    d = 10   # seven digit precision
    if L == 1:
        total = 1.0
    elif n == 0: 
        total = 0
    else:
        bound = int(ceil(2 + sqrt(2 * n * d * log(10))))  # using equation (38)
        for k in range(1, bound + 1):
            b = (n - k + 1) / n * b
            total += b
        old_sum = 1.0
        for j in range(3, L + 1):
            new_sum = total + (n * old_sum) / (j - 2)
            old_sum = total
            total = new_sum
    return total

def universal_code_integers(value):
    """ computes the universal code of integers 
    """
    const =  2.865064
    logsum = log(const,2)
    cond = True # condition
    if value == 0:
        logsum = 0
    else:
        while cond: # Recursive log
            value = log(value,2)
            cond = value > 0.000001
            if value < 0.000001: 
                break
            logsum += value
    return logsum

compute_length_data={
	'nominal':length_data_nominal,
	'numeric':length_data_numeric,
};

compute_length_model={
	'nominal':delta_data_nominal,
	'numeric':delta_data_numeric,
};

delta_data_const={
	'nominal':delta_data_const_nominal,
	'numeric':delta_data_const_numeric,
};
compute_delta_data={
	'nominal':delta_data_nominal,
	'numeric':delta_data_numeric,
};

        
def delta_score(model,antecedent,statistic,usage_total):
    """ adds together the delta model and delta data length  
    """
    if usage_total > 2:
        gain_data = compute_delta_data[model.target_type](model,statistic)
        gain_model = delta_model(model,antecedent)
        if model.gain == "normalized":
            score = (gain_data + gain_model)/usage_total # normalized        
        elif model.gain == "absolute":
            score = (gain_data + gain_model) # absolute
        else:
            print("WRONG gain selected")
    else:
        gain_data = np.NINF
        gain_model = np.NINF
        score = np.NINF 
    return score,gain_data,gain_model

def delta_score_normalized(model,antecedent,statistic,usage_total):
    """ adds together the delta model and delta data length  
    """
    if usage_total > 2:
        gain_data = compute_delta_data[model.target_type](model,statistic)
        gain_model = delta_model(model,antecedent)
        score = (gain_data + gain_model)/usage_total # normalized
    else:
        gain_data = -50
        gain_model = -50
        score = -50 # negative random number because no negative score is selected
    return score,gain_data,gain_model 

def delta_score_absolute(model,antecedent,statistic,usage_total):
    """ adds together the delta model and delta data length  
    """
    if usage_total > 2:
        gain_data = compute_delta_data[model.target_type](model,statistic)
        gain_model = delta_model(model,antecedent)
        score = (gain_data + gain_model) # absolute
    else:
        gain_data = -50
        gain_model = -50
        score = -50 # negative random number because no negative score is selected
    return score,gain_data,gain_model         
        
def compute_length_model(model):
    """ computes code length of the model encoding using 
    1. Universal code of integers for number of rules
    2. Universal code of integers for number of variables in a rule
    3. Uniform code for the number of operations in a certain variable
    4. Uniform code for the pair of variables encoding (combinatorial)    
    """ 
    n_rules = model.number_rules
    l_rules = model.l_universal[n_rules] # empty rule not counted!
    l_pat_len = 0 # number of patterns
    l_pat_comb = 0 
    l_var_type = 0 # variable encoding
    for antecedent in model.antecedent_raw:
        l_pat_len += model.l_universal[len(antecedent)]
        l_pat_comb += model.l_comb[len(antecedent)]
        l_var_type += sum([model.l_var[item[0]] for item in antecedent])
    lm = l_rules + l_pat_len +l_pat_comb + l_var_type 
    return lm

def delta_model(model,antecedent):
    """ computes the delta model code length of adding a rule to the model 
    1. Universal code of integers for number of rules
    2. Universal code of integers for number of variables in a rule
    3. Uniform code for the number of operations in a certain variable
    4. Uniform code for the pair of variables encoding (combinatorial)    
    """ 
    n_rules =model.number_rules+1
    l_rules = model.l_universal[n_rules-1] - model.l_universal[n_rules] # we do not count the empty rule! 
    l_pat_len = -model.l_universal[len(antecedent)]
    l_pat_comb = -model.l_comb[len(antecedent)]
    l_var_type =-sum([model.l_var[item[0]] for item in antecedent])
    dlm = l_rules + l_pat_len + l_pat_comb + l_var_type
    return dlm        
        
       
#length_model={
#	'nominal':length_model_nominal,
#	'numeric':length_model_numeric,
#};
#
#delta_model={
#	'nominal':delta_model_nominal,
#	'numeric':delta_model_numeric,
#};        
        
