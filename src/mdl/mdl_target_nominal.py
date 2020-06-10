# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:44:26 2020

@author: gathu
"""
from src.util.general_functions import log2_0



def length_data_nominal(model):
    """ Computes the length of the data L(D|M)
    """    
    nr = model.number_rules # number of rules
    cl = model.class_codes
    ld = 0
    return ld

def delta_data_nominal(model,statistic):
    """ Computes the delta code length of the data \Delta L(D|M) using NML code
    of adding one rule to the model
    """    
    dld= 0
    return dld

def delta_data_nominal_independent(model,statistic):
    """ Computes the delta code length of the data \Delta L(D|M) using NML code
    of adding one rule to the model
    """    

    dld = 0
    return dld
def delta_data_const_nominal(model):
    """ Computes the constant related with L(D|M) and \Delta L(D|M) which is 
    equal to the length of the data covered by the default rule and encoded 
    by this - L(D default rule|default rule). This is a constant in the greedy
    adding of one rule \Delta L(D|M)
    """      
    constnml = 0
    return constnml
