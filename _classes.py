# -*- coding: utf-8 -*-

from time import time

from src._mdl_ssd import _fit_list
from src.model_predictions import predict_numeric,estimate_weigthedkullbackleibler_gaussian

class SSDC():
    """Subgroup list discovery for numeric targets with an MDL formulation.
    It resorts to greedy and beam search to find the the subgroup list that
    best fits the data
    Parameters
    ----------
    target_type : string, mandatory 
        (possible values: "numeric" or "nominal")
        choose the appropriate target_type (no default value) for the type of 
        rule/subgroup search.
    max_depth : int, optional (default=4) 
        defines the maximum size that subgroup description can take based 
        on the number of variables that the beam search accepts to refine. 
        For example, if 'max_depth = 4' the maximum size of a pattern found is 
        4.
    beam_width : int, optional (default=100)
        defines the width of the beam in the beam search, i.e., the number of
        patterns that are selected at each iteration to be expanded.
   iterative_beam_width
   n_cutpoints : int, optional (default=5)
       number of cut points used to discretize a numeric attribute/variable.
       Note 1: this algorithm creates for each cutpoint a binary split, and 
       the combination of all cutpoints. As an example of the former, if the 
       cut point is x_cut = 5, it will create both the condition x<5 and x>5.
       In relation to the latter, if two of the cut points are x_cut1=3, and
       x_cut2=5, it will also create  3<x<5.
   task : string (default = "discovery") 
       (possible values: "discovery" or "prediction")
       - "discovery": performs subgroup discovery by assuming the last rule of 
       the model as a constant rule and equal to the dataset distribution.
       - "prediction": finds a rule list for prediction by assuming that the
       last rule changes with other rules in the dataset.
   discretization : string (default="static") 
       (possible values: "static" or "dynamic")
       - "static" - performs a priori discretization of all numeric variables
       - "dynamic" - at each iteration of the beam search it conditionally
       discretizes all numeric variables based on the given pattern.
   max_rules : int, optional (default=0)
       Maximum number of subgroups/rules to mine. If max_rules=0 is given it
       continues finding subgroups/rules until no more compression is achieved.
   gain : int, optional (default="normalized")
       (possible values: "absolute" or "normalized")
       Type of score used to expand the beam search and to add a rule/subgroup
       at each iteration.
       - "absolute" - adds the rule/subgroup at each iteration that maximizes
       the normalized gain, i.e., that difference between the length of the 
       existing model minus the length of that model with the candidate 
       subgroup added.
       - "normalized" - adds the rule/subgroup at each iteration that maximizes
       the "absolute" gain normalized by the number of instances covered 
       (usage) by that rule/subgroup.
     Attributes
    ----------
    number_rules: int
        Number of rules of the list excluding the default rule.
    antecedent_description: list of strings
        String of each rule antecedent description.
    consequent_description: list of strings
        String of each rule consequents.
    """

    def __init__(self,target_type="numeric",max_depth=4, beam_width = 100,
                 iterative_beam_width=1,n_cutpoints = 5, task = "discovery",
                 discretization = "static",max_rules = 0,
                 gain = "normalized"):
        self.target_type = target_type
        self.gain = gain
        self.max_depth = max_depth 
        self.beam_width = beam_width
        self.iterative_beam_width= iterative_beam_width
        self.n_cutpoints = n_cutpoints
        self.discretization = discretization
        self.task = task
        self.number_rules = 0
        self.max_rules = max_rules
    
    #TODO:  def __repr__
    def __str__(self): 
        if self.number_rules == 0:
            text2print = "There are not rules"
        else:
            text2print = "" 
            for nr,ant in enumerate(self.antecedent_description):
                if nr == 0:
                    text2print += "IF x in "
                else: 
                    text2print += "ELSE IF x in "
                text2print += ant + " THEN " + self.consequent_description[nr] +\
                            " \n"
            text2print += "ELSE " + self.consequent_lastrule_description
                
                
        return text2print

    def fit(self,df):
        """Fit the model according to the given training data.
        Parameters
        ----------
        df : pandas dataframe with name variables with last column as target 
        variable.
        Returns
        -------
        self : object
        """

        #        if self.min_support_class < self.min_support_global:
        #            warnings.warn("min_support_global > min_support_class => will not"+
        #                          +" be used",UserWarning)
        #
        #        if self.generate_method not in {"FPM","beam","dependence"}:
        #            raise ValueError("Generative method of itemsets shoud be in "\
        #                             "FPM or beam; got (C=%s)"
        #                             % self.generate_method)
        start_time = time()
        rulelist = _fit_list(
                df, self.target_type, self.max_depth,self.beam_width,
                self.iterative_beam_width, self.n_cutpoints,
                self.task,self.discretization,self.max_rules,self.gain)
        self.runtime = time() - start_time
        self.number_rules = rulelist.number_rules
        self.target_type_specific(rulelist)
        self.antecedent_raw = rulelist.antecedent_raw
        self.antecedent_description = rulelist.antecedent_description
        self.consequent_description = rulelist.consequent_description
        self.consequent_lastrule_description = rulelist.consequent_lastrule_description
        self.pattern4prediction = rulelist.pattern4prediction
        self.statistics = rulelist.statistic_rules
        self.default_statistic = rulelist.default_statistic
        self.rule_sets = [[ix for ix,x in enumerate(reversed(bin(bitset)[2:])) if x == '1'] 
                                                for bitset in rulelist.bitset_rules]
        self.length_model = rulelist.length_model
        self.length_data = rulelist.length_data
        self.length_final = self.length_model + self.length_data
        self.length_original = rulelist.length_original
        self.length_ratio = rulelist.length_ratio
        self.measures = rulelist.measures
        self.measures["runtime"] = self.runtime
        return self

     
        
    def predict(self,X):
        """ Predict for new data what is it going to be the performance
        if rule list not fit it does not work
        ----------
        X : a numpy array or pandas dataframe with the variables in the same 
            poistion (column number) as given in "fit" function.
        
        Returns a numpy array y with the predicted values according to the 
        fitted rule list (obtained using the "fit" function above). y has the
        same length as X.shape[0] (number of rows).
        -------
        self : object
        """
        y , usageperrule= predict_numeric(self, X)
        values = X.iloc[:,-1]
        return y,usageperrule

    def target_type_specific(self,rulelist):
        if self.target_type == "nominal":
            self.class_codes = rulelist.class_codes
            self.class_counts = rulelist.class_counts
            self.class_orig = rulelist.class_orig
            self.support_uncovered = rulelist.support_uncovered
            self.usage_rules = rulelist.usage_rules
            self.support_rules = rulelist.support_rules
        elif self.target_type == "numeric":
            self.statistic_rules = rulelist.statistic_rules
            self.default_statistic = rulelist.default_statistic
            self.support_covered = rulelist.support_uncovered
        return self
    
    def rulelist_description(self):
        text2add = "" 
        for nr,ant in enumerate(self.antecedent_description):
            if nr == 0:
                text2add += "IF "
            else: 
                text2add += "ELSE IF "
            text2add += ant + " THEN " + self.consequent_description[nr] +\
            " \n"