# -*- coding: utf-8 -*-

from src.util._read_dataset import transform_dataset
from src.iterative_ruleset_search import find_rulelist
       
def _fit_list(df,target_type, max_depth,beam_width,iterative_beam_width,
              n_cutpoints,task,discretization,max_rules,gain):
    """ this function finds the rule/subgroup list given the selected data,
    target, target type using the Minimum Description Length (MDL) principle 
    formulation.
    """
    # transform data to appropriate format to be used
    data,attributes,target,types,tid_bitsets = transform_dataset(df,
                                            target_type,n_cutpoints,
                                            discretization)
    if max_depth > len(attributes):
        max_depth = len(attributes)
    # search for the rule set 
    rulelist = find_rulelist(data,attributes,target,types,tid_bitsets,task,
                             max_depth,beam_width,iterative_beam_width,
                             max_rules,gain)
    return rulelist