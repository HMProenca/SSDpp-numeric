# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def predict_numeric(model,X):
    dim = X.shape
    if len(dim) == 1: #in case it is only one instance
        y = np.empty(1,dtype=float)
    else:
        y = np.empty(dim[0],dtype=float)
        
    if isinstance(X, pd.DataFrame):
        X = X.values   
    usageperrule = np.zeros(model.number_rules+1)
    for ix,x in enumerate(X):
        for nr in range(model.number_rules):
            decision = decision_pattern(model.pattern4prediction[nr],x)
            if decision:
                y[ix] = model.statistic_rules[nr]["mean"]
                usageperrule[nr] += 1
                break
        else: # default rule
            y[ix] = model.default_statistic["mean"]
            usageperrule[model.number_rules] += 1
    return y,usageperrule     

def decision_pattern(pattern,x):
   decision = True
   for nit in range(pattern["nitems"]):
       type = pattern["type"][nit]
       column = pattern["column"][nit]
       subset = pattern["subset"][nit] 
       decision &=  belongingtest[type](column,subset,x)
   return decision

def belongingtest_numeric(column,subset,x):
    if subset[0] == "minvalue":
        partial_decision = (x[column] >= subset[1])
    elif  subset[0] == "maxvalue":
        partial_decision = (x[column] <= subset[1])
    elif  subset[0] == "interval":
        partial_decision = (x[column] >= subset[1][0]) & (x[column] <= subset[1][1])    
    else:
        print("Wrong terms for belongingtest_numeric function ")
    return partial_decision

def belongingtest_binary(column,subset,x):
    partial_decision = x[column] == subset
    return partial_decision

belongingtest ={
"numeric": belongingtest_numeric,
"binary": belongingtest_binary,
"nominal": belongingtest_binary
}
    
from math import pi,exp
from src.util.general_functions import log2_0 

def gaussian_density(value,var,mean):
    prob = 1/(2*pi*var)**0.5*exp(-(value-mean)**2/var/2)
    return prob

def kullback1(value,mean,var):
    k = 0.5*log2_0(var)+0.5*(value-mean)**2/var*log2_0(exp(1))
    return k


def kullbackleibler(value,mean1,var1,mean2,var2):
    k1 = kullback1(value,mean1,var1)
    k2 = kullback1(value,mean2,var2)
    kl = k2-k1
    return kl
    
#def kullbackleibler(value,mean1,var1,mean2,var2):
#    dens1 = gaussian_density(value,var1,mean1)
#    dens2 = gaussian_density(value,var2,mean2)
#    prob1= dens1/(dens1+dens2)
#    prob2= dens2/(dens1+dens2)
#    kl = prob1*log2_0(prob1/prob2)
#    return kl

def estimate_weigthedkullbackleibler_gaussian(model,X,y):
    kl = 0        
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.values
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values
    kl_perrule = np.zeros(model.number_rules)     
    for ix,x in enumerate(X):
        for nr in range(model.number_rules):
            decision = decision_pattern(model.pattern4prediction[nr],x)
            if decision:
                mean1 = model.statistic_rules[nr]["mean"]
                var1 = model.statistic_rules[nr]["variance"]
                mean2 = model.default_statistic["mean"]
                var2 = model.default_statistic["variance"]
                kl += kullbackleibler(y[ix],mean1,var1,mean2,var2)
                kl_perrule[nr] += kullbackleibler(y[ix],mean1,var1,mean2,var2)
                break
    return kl,kl_perrule

            
