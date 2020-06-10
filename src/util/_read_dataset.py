# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:09:00 2019

@author: gathu
"""
from gmpy2 import mpz,bit_mask
import numpy as np
from pandas.api.types import is_numeric_dtype

def transform_dataset(dforig,target_type,ncutpoints,disc_type):
    # attributes
    attributes = dict()
    df = dforig.copy()
    att_names= df.columns
    nrows = df.shape[0] 
    types = {type: []   for type in ["numeric","nominal","binary","ordinal"]}
    # initialize input of variables (binary, nominal, numeric) 
    for idx in range(df.shape[1]-1):
        colname = att_names[idx]
        # add original name of variable:
        attributes[idx] = dict()
        attributes[idx]["attribute_name"] = colname
        # add and initialize type of variable
        if is_numeric_dtype(df[colname]):
            init_attribute_numeric(attributes,colname,types,idx,nrows,ncutpoints,disc_type)
        else:
            # it automatically ignores NAs
            df.loc[:,colname] = df.loc[:,colname] .astype('category')
            categories = list(df.loc[:,colname].cat.categories)
            df.loc[:,colname] = df.loc[:,colname].cat.codes #transform this column to codes
            if df[colname].nunique() == 2:
                init_attribute_binary(attributes,colname,types,idx,categories)
            else:
                init_attribute_nominal(attributes,colname,types,idx,categories)
    # initialize target variable
    idx = df.shape[1]-1
    target = init_target[target_type](df,att_names,idx)
 
    # transform dataset to 2 numpy arrays
    data = df.values
    
    # initialize the bisets 
    target,tid_bitsets =  create_bitsets(data,target_type,attributes,target)   
    return data,attributes,target,types,tid_bitsets


def init_attribute_binary(attributes,colname,types,idx,categories,operators = None):
    types["binary"].append(idx)
    attributes[idx] = dict()
    attributes[idx]["type"] = "binary"
    attributes[idx]["attribute_name"] = colname    
    attributes[idx]["n_labels"] = len(categories)
    attributes[idx]["label_orig"] = categories
    attributes[idx]["label_code"] = [int(auxi) for auxi in range(len(categories))]
    
def init_attribute_nominal(attributes,colname,types,idx,categories,operators = None):
    types["nominal"].append(idx)
    attributes[idx] = dict()
    attributes[idx]["type"] = "nominal"
    attributes[idx]["attribute_name"] = colname    
    attributes[idx]["n_labels"] = len(categories)
    attributes[idx]["label_orig"] = categories
    attributes[idx]["label_code"] = [int(auxi) for auxi in range(len(categories))] 

def init_attribute_numeric(attributes,colname,types,idx,nrows,ncutpoints,disc_type):
    attributes[idx] = dict()
    attributes[idx]["type"] = "numeric"
    attributes[idx]["attribute_name"] = colname        
    attributes[idx]["discretization"] = disc_type # or numeric
    attributes[idx]["ncutpoints"] = ncutpoints # or numeric
    attributes[idx]["delta"] = nrows /(ncutpoints+1) # or numeric
    types["numeric"].append(idx)
    #delta = df.shape[0] / ncutpoints
    #dataset[idx]["bitset"]  
 
init_attribute={
	'binary':init_attribute_binary,
	'nominal':init_attribute_nominal,
	'numeric':init_attribute_numeric
};        
    
def init_target_nominal(df,att_names,idx):
    """ Initializes the nominal target, with the information of number of 
    counts, label names, and label transformation to integer
    """
    colname = att_names[idx]
    df.loc[:,colname]  = df[colname].astype('category')
    categories = list(df[colname].cat.categories)
    nlabels = len(categories)
    df.loc[:,colname]  = df[colname].cat.codes #transform this column to codes        
    target = dict()
    target["type"] = "nominal"
    target["attribute_name"] = colname        
    target["n_labels"] = nlabels
    target["label_orig"] = categories
    target["label_code"] = [int(auxi) for auxi in range(len(categories))] 
    unique, counts = np.unique(df[colname], return_counts=True)
    target["counts"] = dict(zip(unique, counts))
    target["n_rows"] = df.shape[0]
    return target

def init_target_numeric(df,att_names,idx):
    target = dict()
    colname = att_names[idx]
    target["type"] = "numeric"
    target["attribute_name"] = colname 
    target["max"] = np.max(df[colname])
    target["min"] = np.min(df[colname])
    target["mean"] = np.mean(df[colname])
    target["variance"] = np.var(df[colname])
    target["n_rows"] = df[colname].shape[0]
    target["values"] = np.double(df[colname].to_numpy(copy=True))
    return target         
        
init_target={
'nominal':init_target_nominal,
'numeric':init_target_numeric
}

def indexes2bitset(vector2transform):
        # returns indexes of the first array cindex = bits to flip
        aux_tid = mpz()
        for ii in vector2transform:
            aux_tid = aux_tid.bit_set(int(ii))
        return mpz(aux_tid)

def init_bitset_binary(data,attributes,i_at,tid_bitsets):
    labels = attributes[i_at]["label_code"]
    attributes[i_at]["category_code"] = []
    for il in labels:
        tid_bitsets[(i_at,il)]= dict()
        vector_category = np.where(data[:,i_at]==il)[0]
        tid_bitsets[(i_at,il)]= indexes2bitset(vector_category)
        #attributes[i_at][(i_at,il)] = att_name + " == " + attributes[i_at]["label_orig"][il]
        attributes[i_at][(i_at,il)] =attributes[i_at]["label_orig"][il]

#def init_bitset_numeric_raw_equalsize(data,attributes,target,i_at,tid_bitsets):
#    idx_sorted = np.argsort(data[:,i_at])
#    att_name = attributes[i_at]["attribute_name"]
#    for ncut in range(1,attributes[i_at]["ncutpoints"]+1):
#        cutpoint = round(ncut*attributes[i_at]["delta"])
#        idx_down = idx_sorted[:cutpoint]
#        idx_up = idx_sorted[cutpoint:]
#        maxval = data[idx_sorted[-1],i_at]
#        minval = data[idx_sorted[0],i_at]
#        tid_bitsets[(i_at,-ncut)] = dict()
#        tid_bitsets[(i_at,ncut)] = dict()
#        cutpointvalue = data[idx_sorted[cutpoint],i_at]
#        tid_bitsets[(i_at,-ncut)] = indexes2bitset(idx_down)
#        attributes[i_at][(i_at,-ncut)] = att_name + " in ["+str(minval) +";"+str(cutpointvalue)+")"
#        tid_bitsets[(i_at,ncut)] = indexes2bitset(idx_up)
#        attributes[i_at][(i_at,ncut)] = att_name + " in ["+str(cutpointvalue) +";"+str(maxval)+"]"
        
        
#def init_bitset_numeric_Equalwidthbins(data,attributes,target,i_at,tid_bitsets):
#    att_name = attributes[i_at]["attribute_name"]
#    idx_sorted = np.argsort(data[:,i_at])
#    bin_counts,bin_edges = np.histogram(data[idx_sorted,i_at],bins=attributes[i_at]["ncutpoints"]+1)
#    index_points = np.unique([sum(bin_counts[:idx]) for idx in range(1,len(bin_counts))])
#    ncutpoints = len(index_points)
#    maxval = data[idx_sorted[-1],i_at]
#    minval = data[idx_sorted[0],i_at]
#    attributes[i_at]["ncutpoints"] = ncutpoints
#    for n_cut in range(1,ncutpoints+1):
#        idx_cutpoint = index_points[n_cut-1]
#        idx_down = idx_sorted[:idx_cutpoint]
#        idx_up = idx_sorted[idx_cutpoint:]        
#        tid_bitsets[(i_at,-n_cut)] = dict()
#        tid_bitsets[(i_at,n_cut)] = dict()
#        val_cutpoint = data[idx_sorted[idx_cutpoint],i_at]
#        tid_bitsets[(i_at,-n_cut)] = indexes2bitset(idx_down)
#        attributes[i_at][(i_at,-n_cut)] = att_name + " in ["+str(minval) +";"+str(val_cutpoint)+")"
#        tid_bitsets[(i_at,n_cut)] = indexes2bitset(idx_up)
#        attributes[i_at][(i_at,n_cut)] = att_name + " in ["+str(val_cutpoint) +";"+str(maxval)+"]"   

def init_bitset_numeric(data,attributes,i_at,tid_bitsets,*index_not_consider):
    # TODO - create a missing data variable in case of NAs :) 
    # note that sorting with NAs sends the NAs to the back to the sorting
    idx_sorted = np.argsort(data[:,i_at])
    idx_sorted = np.delete(idx_sorted, np.argwhere(np.isnan(data[idx_sorted,i_at])))
    if index_not_consider:
        idx_sorted = np.setdiff1d(idx_sorted,index_not_consider,assume_unique=True)
    if len(idx_sorted) < 2: return    
    ncutpoints = attributes[i_at]["ncutpoints"]
    quantiles = [1/(ncutpoints+1)*ncut for ncut in range(0,ncutpoints+2)]
    val_quantiles = np.nanquantile(data[idx_sorted,i_at], quantiles)
    if np.isnan(val_quantiles).any(): return    
    bin_counts,bin_edges = np.histogram(data[idx_sorted,i_at],bins=val_quantiles)
    index_points = np.unique([sum(bin_counts[:idx]) for idx in range(1,len(bin_counts))])
    ncutpoints = len(index_points) # check if the number of ncut is smaller than pretended
    #maxval = data[idx_sorted[-1],i_at]
    #minval = data[idx_sorted[0],i_at]
    attributes[i_at]["ncutpoints"] = ncutpoints
    attributes[i_at]["label_code"] = [-int(auxi) for auxi in range(1,ncutpoints+1)] +\
                                    [int(auxi) for auxi in range(1,ncutpoints+1)]
    for n_cut in range(1,ncutpoints+1):
        idx_cutpoint = index_points[n_cut-1]
        idx_down = idx_sorted[:idx_cutpoint]
        idx_up = idx_sorted[idx_cutpoint:]        
        tid_bitsets[(i_at,-n_cut)] = dict()
        tid_bitsets[(i_at,n_cut)] = dict()
        #val_cutpoint = data[idx_sorted[idx_cutpoint],i_at]
        val_cutpoint_up = data[idx_sorted[idx_cutpoint],i_at]
        val_cutpoint_down = data[idx_sorted[idx_cutpoint-1],i_at]      
        tid_bitsets[(i_at,-n_cut)] = indexes2bitset(idx_down)
        #attributes[i_at][(i_at,-n_cut)] = att_name + " in ["+str(minval) +";"+str(val_cutpoint)+")"
        attributes[i_at][(i_at,-n_cut)] = ["maxvalue",val_cutpoint_down]
        tid_bitsets[(i_at,n_cut)] = indexes2bitset(idx_up)
        #attributes[i_at][(i_at,n_cut)] = att_name + " in ["+str(val_cutpoint) +";"+str(maxval)+"]"
        attributes[i_at][(i_at,n_cut)] = ["minvalue",val_cutpoint_up]
    
    label_interval = []
    for n_cut1 in range(1,ncutpoints+1):
        for n_cut2 in range(n_cut1+1,ncutpoints+1):
            label_interval.append([n_cut1,-n_cut2])
            tid_bitsets[(i_at,n_cut1,-n_cut2)] = tid_bitsets[(i_at,n_cut1)]&tid_bitsets[(i_at,-n_cut2)]
            minval = attributes[i_at][(i_at,n_cut1)][1]
            maxval = attributes[i_at][(i_at,-n_cut2)][1]
            attributes[i_at][(i_at,n_cut1,-n_cut2)] = ["interval",[minval,maxval]]
    attributes[i_at]["label_code"] += label_interval
            

    
#def init_bitset_numeric(data,attributes,i_at,tid_bitsets,*index_not_consider):
#    # TODO - create a missing data variable in case of NAs :) 
#    # note that sorting with NAs sends the NAs to the back to the sorting
#    idx_sorted = np.argsort(data[:,i_at])
#    idx_sorted = np.delete(idx_sorted, np.argwhere(np.isnan(data[idx_sorted,i_at])))
#    if index_not_consider:
#        idx_sorted = np.setdiff1d(idx_sorted,index_not_consider,assume_unique=True)
#    if len(idx_sorted) < 2: return    
#    ncutpoints = attributes[i_at]["ncutpoints"]
#    quantiles = [1/(ncutpoints+1)*ncut for ncut in range(0,ncutpoints+2)]
#    val_quantiles = np.nanquantile(data[idx_sorted,i_at], quantiles)
#    if np.isnan(val_quantiles).any(): return    
#    bin_counts,bin_edges = np.histogram(data[idx_sorted,i_at],bins=val_quantiles)
#    index_points = np.unique([sum(bin_counts[:idx]) for idx in range(1,len(bin_counts))])
#    ncutpoints = len(index_points) # check if the number of ncut is smaller than pretended
#    #maxval = data[idx_sorted[-1],i_at]
#    #minval = data[idx_sorted[0],i_at]
#    attributes[i_at]["ncutpoints"] = ncutpoints
#    attributes[i_at]["label_code"] = [-int(auxi) for auxi in range(1,ncutpoints+1)] +\
#                                    [int(auxi) for auxi in range(1,ncutpoints+1)]
#    for n_cut in range(1,ncutpoints+1):
#        idx_cutpoint = index_points[n_cut-1]
#        idx_down = idx_sorted[:idx_cutpoint]
#        idx_up = idx_sorted[idx_cutpoint:]        
#        tid_bitsets[(i_at,-n_cut)] = dict()
#        tid_bitsets[(i_at,n_cut)] = dict()
#        #val_cutpoint = data[idx_sorted[idx_cutpoint],i_at]
#        val_cutpoint_up = data[idx_sorted[idx_cutpoint],i_at]
#        val_cutpoint_down = data[idx_sorted[idx_cutpoint-1],i_at]      
#        tid_bitsets[(i_at,-n_cut)] = indexes2bitset(idx_down)
#        #attributes[i_at][(i_at,-n_cut)] = att_name + " in ["+str(minval) +";"+str(val_cutpoint)+")"
#        attributes[i_at][(i_at,-n_cut)] = ["maxvalue",val_cutpoint_down]
#        tid_bitsets[(i_at,n_cut)] = indexes2bitset(idx_up)
#        #attributes[i_at][(i_at,n_cut)] = att_name + " in ["+str(val_cutpoint) +";"+str(maxval)+"]"
#        attributes[i_at][(i_at,n_cut)] = ["minvalue",val_cutpoint_up]
       
       
            
init_bitset_variable={
	'binary':init_bitset_binary,
	'nominal':init_bitset_binary,
	'numeric':init_bitset_numeric
};
        
def init_bitset_target_nominal(data,target):
    target["bitset"] = dict()
    for c in target["label_code"]:    
        cl_index = np.where(data[:,-1]==c)[0] 
        target["bitset"][c] = indexes2bitset(cl_index)

def init_bitset_target_numeric(data,target):
    target["bitset"] = bit_mask(target["n_rows"])

init_bitset_target={
	'nominal':init_bitset_target_nominal,
	'numeric':init_bitset_target_numeric
};  
    
# convert the transactions ids to bitsets conditional on the target label
def create_bitsets(data,target_type,attributes,target):
    #tid_bitsets = {c: dict() for c in target["label_code"]}
    tid_bitsets = dict()
    init_bitset_target[target_type](data,target)
    for i_at in attributes:
        type_variable = attributes[i_at]["type"]
        init_bitset_variable[type_variable](data,attributes,i_at,tid_bitsets)
    return target,tid_bitsets


## convert the transactions ids to bitsets conditional on the target label
#def bitsets_per_label(data,attributes,target):
#    #tid_bitsets = {c: dict() for c in target["label_code"]}
#    tid_bitsets = dict()
#    cl_index = [np.where(data[:,-1]==c)[0] for c in target["label_code"]]
#    for i_at in attributes:
#        type_variable = attributes[i_at]["type"]
#        init_bitset[type_variable](data,attributes,cl_index,target,i_at,tid_bitsets)
#    return tid_bitsets
#    
#def indexes2bitset(vector_classes,vector_category):
#        # returns indexes of the first array cindex = bits to flip
#        idx_l_and_c = np.intersect1d(vector_classes, vector_category, assume_unique=True, return_indices=True)[1]
#        aux_tid = mpz()
#        for ii in idx_l_and_c:
#            aux_tid = aux_tid.bit_set(int(ii))
#        return aux_tid
#
#def init_bitset_binary(data,attributes,cl_index,target,i_at,tid_bitsets):
#    labels = attributes[i_at]["label_code"]
#    att_name = attributes[i_at]["attribute_name"]
#    classes = target["label_code"]
#    attributes[i_at]["category_code"] = []
#    for il in labels:
#        tid_bitsets[(i_at,il)]= dict()
#        vector_category = np.where(data[:,i_at]==il)[0]
#        for c in classes:
#            tid_bitsets[(i_at,il)][c]= indexes2bitset(cl_index[c],vector_category)
#        attributes[i_at][(i_at,il)] = att_name + " == " + attributes[i_at]["label_orig"][il]
#
#def init_bitset_numeric(data,attributes,cl_index,target,i_at,tid_bitsets):
#    idx_sorted = np.argsort(data[:,i_at])
#    att_name = attributes[i_at]["attribute_name"]
#    for ncut in range(1,attributes[i_at]["ncutpoints"]+1):
#        cutpoint = round(ncut*attributes[i_at]["delta"])
#        idx_down = idx_sorted[:cutpoint]
#        idx_up = idx_sorted[cutpoint:]
#        maxval = data[idx_sorted[-1],i_at]
#        minval = data[idx_sorted[0],i_at]
#        tid_bitsets[(i_at,-ncut)] = dict()
#        tid_bitsets[(i_at,ncut)] = dict()
#        for c in target["label_code"]:
#            cutpointvalue = data[idx_sorted[cutpoint],i_at]
#            tid_bitsets[(i_at,-ncut)][c] = indexes2bitset(cl_index[c],idx_down)
#            attributes[i_at][(i_at,-ncut)] = att_name + " in ["+str(minval) +";"+str(cutpointvalue)+")"
#            tid_bitsets[(i_at,ncut)][c] = indexes2bitset(cl_index[c],idx_up)
#            attributes[i_at][(i_at,ncut)] = att_name + " in ["+str(cutpointvalue) +";"+str(maxval)+"]"    
    
    
    
