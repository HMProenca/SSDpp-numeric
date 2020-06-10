# -*- coding: utf-8 -*-


import csv
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
import os
import re
import itertools
from subprocess import call,check_output
import shutil
from copy import deepcopy
from time import time
import itertools
from math import log,exp
from numpy import mean
log2 = lambda n: log(n or 1, 2)

from src.util.general_functions import log2_0
from src.util.results2folder import attach_results,print2folder,makefolder_name
filedatasets =  "./data/numeric target/"
delimiter=','
results_file='./results.csv'
depthmax=5.0
def kullbackleibler_gaussian(mean_dataset,variance_dataset,values):
    usage = len(values)
    l_e = log2(exp(1))
    RSS = sum([(val-mean_dataset)**2 for val in values])
    variance = np.var(values)
    kl_aux1 = 0.5*log2(variance_dataset)+\
            0.5*RSS/usage/variance_dataset*l_e
    kl_aux2 = 0.5*log2(variance)+0.5*l_e
    kl = kl_aux1 - kl_aux2 
    wkl = usage*kl
    return kl,wkl

def wracc_numeric(mean_dataset,values):
    usage = len(values)
    meanval = np.mean(values)
    wracc = usage*np.absolute(mean_dataset-meanval)
    return wracc

def discoverymetrics_numeric(targetvalues,nrules,rules_supp,rules_usg,subgroup_sets_support,subgroup_sets_usage):
    mean_dataset= np.mean(targetvalues)
    variance_dataset= np.var(targetvalues)
    kl_supp,kl_usg,wkl_supp,wkl_usg,wkl_sum = np.zeros(nrules),np.zeros(nrules), np.zeros(nrules), np.zeros(nrules), np.zeros(nrules)
    wacc_supp, wacc_usg = np.zeros(nrules),np.zeros(nrules)
    support, usage = np.zeros(nrules),np.zeros(nrules)
    stdrules = np.zeros(nrules)
    top1_std = 0 
    for r in range(nrules):
        idx_support = list(subgroup_sets_support[r])
        values_support  =  targetvalues[idx_support]
        support[r] = len(values_support)
        kl_supp[r], wkl_supp[r] = kullbackleibler_gaussian(mean_dataset,variance_dataset,values_support)
        wacc_supp[r] = wracc_numeric(mean_dataset,values_support)
    for r in range(len(subgroup_sets_usage)):
        idx_usage = list(subgroup_sets_usage[r])
        values_usage =  targetvalues[idx_usage]
        usage[r] = len(values_usage)        
        kl_usg[r], wkl_usg[r] =kullbackleibler_gaussian(mean_dataset,variance_dataset,values_usage)
        wacc_usg[r] = wracc_numeric(mean_dataset,values_usage)
        stdrules[r] = np.std(values_usage)
        if r == 0:
            top1_std =  np.std(values_usage)
    
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
    measures["wkl_sum"] = wkl_sum
    measures["std_rules"] = np.mean(stdrules)
    measures["top1_std"] = top1_std
    return measures

def discovery_itemset(data,model,cl):
    # data = labeled data
    # Initialization of the model
    #[(0->Pattern[s],
    #  1->class[s],
    #  2->L(cl|P)[f],
    #  3->L(not cl|P)[s],
    #  4->Pattern and class[s],
    #  5-> Pr(cl|P)[f],
    #  (...),...]
    # Empty model
    #probClass =[[0]*len(cl)  for i in range(len(undata))]

    rules_supp = {nr: {c: int(0) for c in cl} for nr in model.keys()}
    rules_usg = {nr: {c: int(0) for c in cl} for nr in model.keys()}
    count_cl = {c: int(0) for c in cl}
    pred = []
    prob = []
    RULEactivated = []
    nr = len(model)
    intersect = np.zeros([nr,nr],dtype = np.uint)
    jaccard = np.zeros([nr,nr])
    # Find majority class
    for t in data:
        active_r = list()
        first = True 
        for r in range(nr):
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
        for rr in itertools.combinations(active_r, 2):
            intersect[rr] +=1

    for rr in itertools.combinations(range(nr), 2):
        inter =intersect[rr]
        supp1 = intersect[(rr[0],rr[0])]
        supp2 = intersect[(rr[1],rr[1])]
        jaccard[rr]= inter/(supp1+supp2-inter)  
    
    # remove empty rule column and row
    jaccard = np.delete(jaccard, -1, 0)
    jaccard = np.delete(jaccard, -1, 1)
    # average over all possible cases 
    uptm = np.triu_indices(nr-1,1)
    jacc_avg = np.sum(jaccard)/len(uptm[0])
    jacc_consecutive_avg = np.mean(np.diagonal(jaccard,1))
    avg_supp = np.mean([sum([rules_supp[r][c] for c in cl]) for r in range(nr-1)])
    avg_usg = np.mean([sum([rules_usg[r][c] for c in cl]) for r in range(nr-1)])
    
    return pred, prob, RULEactivated,rules_supp,rules_usg,count_cl,jacc_avg,avg_supp,avg_usg


def read_csvfile(source):
	with open(source, 'r') as csvfile:
		readfile = csv.reader(csvfile, delimiter='\t')
		results=[row for row in readfile if len(row)>0]
	return results

def write_file_conf(listofthings,file2write):
    with open(file2write, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(listofthings)
        
     
def findnumber(item):
    for i in item.split():
        try:
            #trying to convert i to float
            number = float(i)
            #break the loop if i is the first string that's successfully converted
            break
        except:
            continue
    return number


# estimate in unseed data functions
def decision_pattern(pattern,x):
   decision = True
   for nit in range(pattern["nitems"]):
       type = pattern["type"][nit]
       column = pattern["column"][nit]
       subset = pattern["subset"][nit] 
       decision &=  belongingtest[type](column,subset,x)
   return decision

def belongingtest_numeric(column,subset,x):
    if subset[0] == np.NINF:
        partial_decision = (x[column] > subset[0]) & (x[column] < subset[1])
    elif subset[1] == np.inf:
        partial_decision = (x[column] > subset[0]) & (x[column] < subset[1])
    return partial_decision

def belongingtest_binary(column,subset,x):
    partial_decision = x[column] == subset
    return partial_decision

belongingtest ={
"numeric": belongingtest_numeric,
"binary": belongingtest_binary,
"nominal": belongingtest_binary
}
def kullback1(value,meanval,var):
    k = 0.5*log2_0(var)+0.5*(value-meanval)**2/var*log2_0(exp(1))
    return k

def kullbackleibler(value,mean1,var1,mean2,var2):
    k1 = kullback1(value,mean1,var1)
    k2 = kullback1(value,mean2,var2)
    kl = k2-k1
    return kl

def estimate_weigthedkullbackleibler_gaussian(default_statistic,statistic_rules,pattern4prediction,X,y):
    kl = 0        
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.DataFrame):
        y = y.values
    kl_perrule = np.zeros(len(pattern4prediction)) 
    usg_rule =  np.zeros(len(pattern4prediction))  
    for ix,x in enumerate(X):
        for nr in range(len(pattern4prediction)):
            decision = decision_pattern(pattern4prediction[nr],x)
            if decision:
                mean1 = statistic_rules[nr]["mean"]
                var1 = statistic_rules[nr]["variance"]
                mean2 = default_statistic["mean"]
                var2 = default_statistic["variance"]
                kl += kullbackleibler(y[ix],mean1,var1,mean2,var2)
                kl_perrule[nr] += kullbackleibler(y[ix],mean1,var1,mean2,var2)
                usg_rule[nr] += 1
                break
    return kl,kl_perrule,usg_rule

#######################################################################################
#  DSSD
##############################################  
algorithmname = "topk"
datasetnames = ["cholesterol","baseball","autoMPG8","dee","ele-1","forestFires","concrete",\
                "treasury","wizmir","abalone","puma32h","ailerons","elevators",\
                "bikesharing","california","house"]

nrulesssd = {"cholesterol":1,"baseball":8,"autoMPG8":10,"dee":8,"ele-1":9,"forestFires":23,
"concrete":19,"treasury":31,"wizmir":22,"abalone":25,"puma32h":42,"ailerons":197,"elevators":160,
"bikesharing":127,"california":163,"house":280}

savefile = makefolder_name(algorithmname)
savefile = savefile + "/summary.csv"
#savefile = "./results/"+algorithmname+"a_summary.txt"
print("datasetname,kl_supp,avg_supp,wkl_supp,kl_usg,avg_usg,wkl_usg,wacc_supp,wacc_usg,wkl_sum,jacc_avg,n_rules,avg_items,nrows_train,std_rules,top1_std,runtime,",file=open(savefile, "w"))
testpercentage = 0.2
beam_width = 100
depthmax = 5
topkalgorithm = 2000
for datasetname in datasetnames:
    print("dataset name : " + str(datasetname))
    file_data =  filedatasets+datasetname+".csv"
    df = pd.read_csv(file_data,sep =delimiter)
    df.rename(columns={ df.columns[-1]: "class" }, inplace = True)
    dfaux = df
    #dfaux, test = train_test_split(df, test_size=testpercentage,random_state = 1)
    dataset = dfaux.to_dict(orient ="records")
    new_dataset=deepcopy(dataset)   
    # change configuration file of DSSD
    conf_file=read_csvfile('./otheralgorithms/DSSD/bin/tmpModeltopk.conf')
    conf_file[10]=['topK = '+str(int(topkalgorithm))]
    nrows = df.shape[0]
    if nrows > 2000:
        conf_file[14]=['searchType = '+ "beam"]
    else:
        conf_file[14]=['searchType = '+ "dfs"]  
    #conf_file[12]=['postSelect = '+str(int(top_k))]
    conf_file[19]=['beamWidth = '+str(int(beam_width))]
    conf_file[15]=['maxDepth = '+str(min(int(depthmax),10))]
    
    
    write_file_conf(conf_file,'./otheralgorithms/DSSD/bin/tmp.conf')
    
    # preprocessing dataset info
    columnames = dfaux.columns[:-1]
    typevar = ["numeric" if is_numeric_dtype(dfaux[col]) else "nominal" for col in columnames]
    limits = []
    for icol,col in enumerate(columnames):
        if typevar[icol] == "numeric":
            minval = min(dfaux[col])
            maxval = max(dfaux[col])
            limits.append([minval,maxval])
        elif typevar[icol] == "nominal":
            categories = np.unique(dfaux[col])
            limits.append(categories)
        else:
            print("went wrong")
    
    # check if path exists
    if not os.path.exists('.//otheralgorithms//DSSD//xps//dssd'):
    	os.makedirs('.//otheralgorithms//DSSD//xps//dssd')
    else:
    	if False:
    		shutil.rmtree('.//otheralgorithms//DSSD//xps//dssd')
    		os.makedirs('.//otheralgorithms//DSSD//xps//dssd')
    # prepare dataset to be read 
    dfaux.to_csv("./otheralgorithms/DSSD/data/datasets/tmp/tmp.csv",index=False)    
    call(["csv2arff", "./otheralgorithms/DSSD/data/datasets/tmp/tmp.csv",
          "./otheralgorithms/DSSD/data/datasets/tmp/tmp.arff"])    
    # run DSSD
    timespent=time()
    os.chdir("./otheralgorithms/DSSD/bin")
    call(["emc64-mt-modified.exe"])
    os.chdir("../../../")
    timespent=time()-timespent
    os.remove("./otheralgorithms/DSSD/data/datasets/tmp/tmp.csv") 
    os.remove("./otheralgorithms/DSSD/data/datasets/tmp/tmp.arff") 
    
    # read output files
    auxfiles = [path for path in os.listdir('./otheralgorithms/DSSD/xps/dssd/')]
    generated_xp='./otheralgorithms/DSSD/xps/dssd/'+auxfiles[-1] # last one
    timestamp=generated_xp.split('-')[1]
    # find transaction ids of subgroups
    generated_xp_subsets_path=generated_xp+'/subsets'
    all_generated_subgroups_files=[generated_xp_subsets_path+'/'+x
                                   for x in os.listdir(generated_xp_subsets_path)]
    # find descriptions of subgroups    
    description_files=generated_xp+'/'+"stats2-" +timestamp+".csv"
    
    #count number of items per subgroup
    descriptions = read_csvfile(description_files)
    top_k = nrulesssd[datasetname]
    if len(descriptions)-1 > top_k:
        descriptions =  descriptions[:top_k+1]
        all_generated_subgroups_files =  all_generated_subgroups_files[:top_k]
        

    nitems = []
    pattern4prediction = []
    for row in descriptions[1:]:
        #count items
        nitems.append(1+row[0].count("&&")) 
        # find pattern descritpion
        subsetdefinition = {"type":[],"var_name": [], 
                            "subset":[], 
                            "column":  [],"nitems" : 0}
        rowsplit = row[0].split(";")
        pattern = rowsplit[-1].split("&&")
        #print(description)
        for item in pattern:
            for icol,col in enumerate(columnames):
                if re.search(col+"\s", item):
                    if typevar[icol] == "numeric":
                        number = findnumber(item)
                        if ">" in item:
                            subset = [number,np.inf]
                        elif "<" in item:
                            subset = [np.NINF,number]
                        else:
                            print("went wrong")
                    elif typevar[icol] == "nominal":
                        for cat in limits[icol]:
                            if cat in item:
                                subset = cat
                    else:
                        print("went wrong")
                    subsetdefinition["type"].append(typevar[icol])
                    subsetdefinition["var_name"].append(col)
                    subsetdefinition["subset"].append(subset)
                    subsetdefinition["column"].append(icol)
                    subsetdefinition["nitems"] += 1
        pattern4prediction.append(subsetdefinition)
                    
    subgroup_sets_support=[]
    support_union=set()
    nb_subgroups=0
    rules_supp = []
    #if len(all_generated_subgroups_files) <= top_k:
    #    pass
    #elif len(all_generated_subgroups_files) > top_k:
    #    all_generated_subgroups_files = all_generated_subgroups_files[:top_k]
    for subgroup_file in all_generated_subgroups_files:
        aux_subgroup=read_csvfile(subgroup_file)[2:]
        subgroup_biset=[row[0] for row in aux_subgroup]
        subgroup_index = set(i for i,x in enumerate(subgroup_biset) if x=='1')
        subgroup_sets_support.append(subgroup_index)
        support = len(subgroup_index)
        rules_supp.append(support)
        nb_subgroups+=1

    rules_usg = []
    subgroup_sets_usage= []
    statistic_rules = []
    target_values = dfaux["class"].to_numpy()
    for r in range(len(subgroup_sets_support)):
        previous_sets = [subgroup_sets_support[ii] for ii in range(r)]
        auxset= subgroup_sets_support[r].difference(*previous_sets)
        values = target_values[list(auxset)]
        #print(len(values))
        if len(values)>2:
            rulestat = {"mean":np.mean(values), "variance": np.var(values)}
        else:
            rulestat = {"mean":np.nan, "variance": np.nan}
        statistic_rules.append(rulestat)
        usage = len(auxset)
        subgroup_sets_usage.append(auxset)     
        rules_usg.append(usage)
        
    # remove zero cases:
    toremove= []
    for nr in range(len(rules_usg)):
        if rules_usg[nr] <3:
            toremove.append(nr)
    rules_usg = [val for nr,val in enumerate(rules_usg) if nr not in toremove]
    subgroup_sets_usage  = [val for nr,val in enumerate(subgroup_sets_usage) if nr not in toremove]
    statistic_rules= [val for nr,val in enumerate(statistic_rules) if nr not in toremove]
    nitems = [val for nr,val in enumerate(nitems) if nr not in toremove]
    pattern4prediction = [val for nr,val in enumerate(pattern4prediction) if nr not in toremove]   
    
    nrules = len(rules_supp)
    intersect = np.zeros([nrules,nrules],dtype = np.uint) 
    jaccard = np.zeros([nrules,nrules])
    for kk in range(nrules):
        intersect[kk,kk] = len(subgroup_sets_support[kk])
        for kk2 in range(kk+1,nrules):
            intersect[kk,kk2] = len(subgroup_sets_support[kk] & subgroup_sets_support[kk2])
    
    for rr in itertools.combinations(range(nrules), 2):
        inter =intersect[rr]
        supp1 = intersect[(rr[0],rr[0])]
        supp2 = intersect[(rr[1],rr[1])]
        jaccard[rr]= inter/(supp1+supp2-inter)  
        
    # average over all possible cases 
    uptm = np.triu_indices(nrules,1)
    jacc_avg = np.sum(jaccard)/len(uptm[0])
    jacc_consecutive_avg = np.mean(np.diagonal(jaccard,1))
    avg_supp = np.mean(rules_supp)
    avg_usg= np.mean(rules_usg)

    # drop rules with zero usage
    targetvalues = dfaux["class"].to_numpy()
    measures  =  discoverymetrics_numeric(targetvalues,nrules,rules_supp,rules_usg,
                                          subgroup_sets_support,subgroup_sets_usage)
    measures["jacc_avg"] = jacc_avg
    measures["n_rules"] = nrules
    measures["avg_items"] = np.mean(nitems)    
    # prediction on holdoutdatay= 
    default_statistic = {"mean": np.mean(targetvalues), "variance": np.var(targetvalues)}
    #yvalues = test["class"].to_numpy()
    #X =  test.loc[:, test.columns != 'class']
    #wkl_sum_test,kl_perrule,usg_rule2 = estimate_weigthedkullbackleibler_gaussian(default_statistic,
    #                                statistic_rules,pattern4prediction,X,yvalues)
    number_samples_train = len(targetvalues)
    #number_samples_test = len(yvalues)
    
    #confirmwkl,asda,usg_confirm = estimate_weigthedkullbackleibler_gaussian(default_statistic,
    #                                statistic_rules,pattern4prediction,dfaux,targetvalues)    
    measures["number_samples_train"] = number_samples_train
    print("real wkl: " + str(measures["wkl_sum"]))   

    
    print("%s , %.4f , %.4f , %.4f , %.4f,\
          %.4f , %.4f , %.4f , %.4f  ,\
          %.4f , %.4f , %.4f  , %.4f ,\
          %.4f , %.4f,  %.4f , %.4f"\
              %(datasetname,\
                measures["kl_supp"],\
                measures["avg_supp"],\
                measures["wkl_supp"],\
                measures["kl_usg"],\
                measures["avg_usg"],\
                measures["wkl_usg"],\
                measures["wacc_supp"],\
                measures["wacc_usg"],\
                measures["wkl_sum"],\
                measures["jacc_avg"],\
                measures["n_rules"],\
                measures["avg_items"],
                measures["number_samples_train"],\
                measures["std_rules"],\
                measures["top1_std"],\
                timespent),\
                file=open(savefile, "a"))