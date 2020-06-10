# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:10:10 2019

@author: gathu
"""
import numpy as np
from math import pi,exp
from itertools import combinations
from gmpy2 import xmpz,mpz,popcount

from src.util.general_functions import log2_0 

def kullbackleibler_gaussian_paramters(model,values):
    usage = len(values)
    RSS = sum([(val-model.default_statistic["mean"])**2 for val in values])
    variance = np.var(values)
    #print("variance" + str(variance))
    if usage:
        kl_aux1 = 0.5*log2_0(model.default_statistic["variance"])+\
                0.5*RSS/usage/model.default_statistic["variance"]*model.l_e
        kl_aux2 = 0.5*log2_0(variance)+0.5*model.l_e
        kl = kl_aux1 - kl_aux2 
        wkl = usage*kl
    else:
        kl = 0
        wkl = 0
    return kl,wkl

def kullback1(value,mean,var):
    k = 0.5*log2_0(var)+(value-mean)**2/(2*var)*log2_0(exp(1))
    return k

def gaussian_density(value,var,mean):
    return 1/(2*pi*var)**0.5*exp(-(value-mean)**2/var/2)

def kullbackleibler(value,mean1,var1,mean2,var2):
    prob1 = gaussian_density(value,var1,mean1)
    prob2 = gaussian_density(value,var2,mean2)
    kl = prob1*log2_0(prob1/prob2)
    return kl

def estimate_weigthedkullbackleibler_gaussian(model,X,y):
    kl = 0        
    if isinstance(X, pd.DataFrame):
        X = X.values   
    for ix,x in enumerate(X):
        for nr in range(model.number_rules):
            decision = decision_pattern(model.pattern4prediction[nr],x)
            if decision:
                mean1 = model.statistic_rules[nr]["mean"]
                var1 = model.statistic_rules[nr]["variance"]
                mean2 = model.default_statistic[nr]["mean"]
                var2 = model.default_statistic[nr]["mean"]
                kl += kullbackleibler(x,mean1,var1,mean2,var2)
                break
    return kl



def wracc_numeric(model,values):
    usage = len(values)
    mean = np.mean(values)
    wracc = usage*np.absolute(model.default_statistic["mean"]-mean)
    return wracc
    

def numeric_discovery_measures(model):
    nrules= model.number_rules
    #nrows= len(model.target_values)
    kl_supp,kl_usg,wkl_supp,wkl_usg,wkl_sum = np.zeros(nrules),np.zeros(nrules), np.zeros(nrules), np.zeros(nrules), np.zeros(nrules)
    wacc_supp, wacc_usg = np.zeros(nrules),np.zeros(nrules)
    support, usage = np.zeros(nrules),np.zeros(nrules)
    std_rules = [stat["variance"]**0.5 for stat in model.statistic_rules]
    std_rulesalternative = []
    tid_covered =  mpz()
    for r in range(nrules):
        tid_support = model.bitset_rules[r]
        tid_usage = model.bitset_rules[r] &~ tid_covered
        tid_covered = tid_covered | tid_support
        aux_bitset = xmpz(tid_support)
        idx_bits = list(aux_bitset.iter_set())
        values_support  =  model.target_values[idx_bits]
        aux_bitset = xmpz(tid_usage)
        idx_bits = list(aux_bitset.iter_set())
        values_usage = model.target_values[idx_bits]
        support[r] = len(values_support)
        usage[r] = len(values_usage)        
        kl_supp[r], wkl_supp[r] = kullbackleibler_gaussian_paramters(model,values_support)
        kl_usg[r], wkl_usg[r] = kullbackleibler_gaussian_paramters(model,values_usage)
        std_rulesalternative.append(np.std(values_usage))
        wacc_supp[r] = wracc_numeric(model,values_support)
        wacc_usg[r] = wracc_numeric(model,values_usage)
        #print(wkl_usg)
    
    
    wkl_sum = sum(wkl_usg)
    #  Average them all!!!!
    measures = dict()
    measures["avg_supp"] = np.mean(support)
    measures["kl_supp"] = np.mean(kl_supp)
    measures["wkl_supp"]  = np.mean(wkl_supp)

    measures["avg_usg"] = np.mean(usage)
    measures["kl_usg"]  = np.mean(kl_usg)
    measures["wkl_usg"]  = np.mean(wkl_usg)
    
    measures["wacc_supp"]  = np.mean(wacc_supp)
    measures["wacc_usg"]   = np.mean(wacc_usg)
    
    
    uptm = np.triu_indices(nrules-1,1)
    measures["jacc_avg"] = np.sum(model.jaccard_matrix)/len(uptm[0])
    measures["n_rules"] = model.number_rules
    measures["avg_items"] = sum([len(ant) for ant in model.antecedent_raw])/model.number_rules
    measures["wkl_sum"] = wkl_sum
    measures["std_rules"] = np.mean(std_rules)
    measures["top1_std"] =std_rules[0]
   
    measures["length_orig"] = model.length_original
    measures["length_final"] = model.length_data + model.length_model
    measures["length_ratio"] = model.length_ratio
    
    return measures 

          
def nominal_discovery_measures(model):
    cl= model.class_codes
    nrules= model.number_rules
    nrows= sum([model.class_counts[c] for c in cl])
    prob_default = [model.class_counts[c]/nrows for c in cl]
    kl_supp,kl_usg,wkl_supp,wkl_usg,wkl_sum = 0,0,0,0,0
    wacc_supp, wacc_usg = 0,0
    unused_supp,unused_usg  =0,0
    for r in range(nrules):
        # support related kl
        sumt= sum([model.support_rules[r][c] for c in cl])
        if sumt != 0:
            prob_rule = [model.support_rules[r][c]/sumt for c in cl]
            kl_suppaux = sum([prob_rule[ic]*log2_0(prob_rule[ic]/prob_default[ic])
                             for ic,c in enumerate(cl)])
        else:
            unused_supp +=1
            kl_suppaux= 0            
        kl_supp += kl_suppaux
        wkl_supp  += kl_suppaux*sumt

        # support related weighted relative accuracy
        if len(cl) == 2 and sumt != 0:
            acc_dataset = prob_default[0]  
            acc_rule = prob_rule[0]
            wacc_supp += (sumt/nrows)*abs(acc_rule-acc_dataset) if sumt != 0 else 0
        elif len(cl) != 2 and sumt != 0:
            wacc_suppaux= np.zeros(len(cl))
            for ic,c in enumerate(cl):
                acc_dataset = prob_default[ic]
                acc_rule = prob_rule[ic]
                wacc_suppaux[ic] = (sumt/nrows)*abs(acc_rule-acc_dataset)            
            wacc_supp += np.mean(wacc_suppaux)
        # usage related kl
        sumt= sum([model.usage_rules[r][c] for c in cl])
        if sumt != 0:
            prob_rule = [model.usage_rules[r][c]/sumt for c in cl]        
            kl_usgaux = sum([prob_rule[ic]*log2_0(prob_rule[ic]/prob_default[ic])
                             for ic,c in enumerate(cl)])
        else:
            unused_usg +=1
            kl_usgaux= 0
        kl_usg += kl_usgaux
        wkl_usg  += kl_usgaux*sumt
        print(sumt)
        # usage related weighted relative accuracy
        if len(cl) == 2 and sumt != 0:
            acc_dataset = prob_default[0]
            acc_rule = prob_rule[0]
            wacc_usg += (sumt/nrows)*abs(acc_rule-acc_dataset)  
        elif len(cl) != 2 and sumt != 0:
            wacc_usgaux= np.zeros(len(cl))
            for ic,c in enumerate(cl):
                acc_dataset = prob_default[ic]
                acc_rule = prob_rule[ic]
                wacc_usgaux[ic] = (sumt/nrows)*abs(acc_rule-acc_dataset)            
            wacc_usg += np.mean(wacc_usgaux)
        else:
            pass
        
    # WRACC for the union
    supp_union = [sum([model.support_rules[r][c] for r in range(nrules)]) for c in cl]
    sumt= sum([supp_union[c] for c in cl])
    prob_rule = [supp_union[c]/sumt for c in cl]
    if len(cl) == 2 and sumt != 0:
        acc_dataset = prob_default[0]  
        acc_rule = prob_rule[0]
        wacc_union = (sumt/nrows)*abs(acc_rule-acc_dataset) if sumt != 0 else 0
    elif len(cl) != 2 and sumt != 0:
        wacc_suppaux= np.zeros(len(cl))
        for ic,c in enumerate(cl):
            acc_dataset = prob_default[ic]
            acc_rule = prob_rule[ic]
            wacc_suppaux[ic] = (sumt/nrows)*abs(acc_rule-acc_dataset)            
        wacc_union = np.mean(wacc_suppaux)  
        
    #  Average them all!!!!
    measures = dict()
    measures["kl_supp"] = kl_supp/(nrules-unused_supp)
    measures["avg_supp"] = sum([sum([model.support_rules[r][c] for c in cl]) for r in range(nrules)])/nrules
    measures["wkl_supp"]  = wkl_supp/(nrules-unused_supp) 

    measures["kl_usg"]  = kl_usg/(nrules-unused_usg)
    measures["avg_usg"] = sum([sum([model.usage_rules[r][c] for c in cl]) for r in range(nrules)])/nrules
    measures["wkl_usg"]  = wkl_usg/(nrules-unused_usg)
    
    measures["wacc_supp"]  = wacc_supp/(nrules-unused_usg) 
    measures["wacc_usg"]   = wacc_usg/(nrules-unused_usg)
    measures["wacc_union"]   = wacc_union
    
    uptm = np.triu_indices(nrules-1,1)
    measures["jacc_avg"] = np.sum(model.jaccard_matrix)/len(uptm[0])
    measures["n_rules"] = model.number_rules
    measures["avg_items"] = sum([len(ant) for ant in model.antecedent_raw])/model.number_rules
    measures["wkl_sum"] = wkl_usg
    measures["length_orig"] = model.length_original
    measures["length_final"] = model.length_data + model.length_model
    measures["length_ratio"] = model.length_ratio
    
    
    return measures 


def jaccard_index_model(model,tid_bitsets):
    nrules = model.number_rules
    intersect = np.zeros([nrules,nrules],dtype = np.uint)
    for r1 in range(nrules):
        tid_rule1 = model.bitset_rules[r1]
        for r2 in range(nrules):
            tid_rule2 = model.bitset_rules[r2]
            intersect[r1,r2] = popcount(tid_rule1 &  tid_rule2)
    jaccard = np.zeros([nrules,nrules])
    for rr in combinations(range(nrules), 2):
        inter =intersect[rr]
        supp1 = intersect[(rr[0],rr[0])]
        supp2 = intersect[(rr[1],rr[1])]
        jaccard[rr]= inter/(supp1+supp2-inter)  
    #uptm = np.triu_indices(nrules-1,1)
    return jaccard



def discovery_itemset(data,model):
    nrules = model.number_rules
    cl = model.class_codes
    rules_supp = {nr: {c: int(0) for c in cl} for nr in range(nrules)}
    rules_usg = {nr: {c: int(0) for c in cl} for nr in range(nrules)}
    count_cl = {c: int(0) for c in cl}
    pred = []
    prob = []
    RULEactivated = []
    intersect = np.zeros([nrules,nrules],dtype = np.uint)
    jaccard = np.zeros([nrules,nrules])
    # Find majority class
    for t in data:
        active_r = list()
        first = True 
        for r in range(nrules):
            if model[r]['p'] <= t and first:
                pred.append(model[r]['cl'])
                prob.append(model[r][model[r]['cl']])
                RULEactivated.append(r)
                active_r.append(r)
                intersect[r,r] +=1
                for ic, c in enumerate(cl):
                    if c <= t:
                        rules_supp[r][c] +=1
                        rules_usg[r][c] +=1
                        count_cl[c] +=1
                first = False
            elif model[r]['p'] <= t and not first:
                active_r.append(r)
                intersect[r,r] +=1
                for ic, c in enumerate(cl):
                    if c <= t:
                        rules_supp[r][c] +=1 
        for rr in combinations(active_r, 2):
            intersect[rr] +=1

    for rr in combinations(range(nrules), 2):
        inter = intersect[rr]
        supp1 = intersect[(rr[0],rr[0])]
        supp2 = intersect[(rr[1],rr[1])]
        jaccard[rr]= inter/(supp1+supp2-inter)  
    
    # remove empty rule column and row
    jaccard = np.delete(jaccard, -1, 0)
    jaccard = np.delete(jaccard, -1, 1)
    # average over all possible cases 
    uptm = np.triu_indices(nrules-1,1)
    jacc_avg = np.sum(jaccard)/len(uptm[0])
    jacc_consecutive_avg = np.mean(np.diagonal(jaccard,1))
    avg_supp = np.mean([sum([rules_supp[r][c] for c in cl]) for r in range(nr-1)])
    avg_usg = np.mean([sum([rules_usg[r][c] for c in cl]) for r in range(nr-1)])
    
    return pred, prob, RULEactivated,rules_supp,rules_usg,count_cl,jacc_avg,avg_supp,avg_usg