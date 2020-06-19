# Guide to run SSD++ for numeric targets

This code reproduces the experiments of the 2020 ECML-PKDD paper: "Discovering outstanding subgroup lists for numeric targets using MDL". 

## Getting Started

These instructions depict the requirments and format to run the code standalone, i.e., to run the code for new experiments of your own.

### Prerequisites

**Version**: The whole code was developed under Python 3.7.

**Common libraries**: it also requires the following commonly used library packages that automatically come with the Anaconda Python distribution:

* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)- for visualization of the graphs presented in the article.
* [seaborn](https://seaborn.pydata.org/) - for the probability density function of the Hotel dataset presented in the article.

In case that you do not have these already installed with your current python distribution you can always install them using the command pip install in the command line (or conda install in case the Anaconda distribution is present) as follows:

```
pip install numpy pandas scipy matplotlib seaborn
```
or
```
conda install numpy # repeat this for the other packages
```

**Not so common libraries**: In addition to the other packages our code also requires the following less common packages:

* [gmpy2](https://pypi.org/project/gmpy2/) - for multiple precision integers and bitwise computations.
* [numba](http://numba.pydata.org/) - for speeding up calculations of the statistics (available automatically with Anaconda).

and in case you cannot install them directly through pip install (as depicted previously) please consider (if in Windows) the following unofficial list of binaries (for which I take no responsability):

https://www.lfd.uci.edu/~gohlke/pythonlibs/



### Installing

*Download* the folder containing the whole code, *change directory* to the folder (at the same level as _classes.py), and *run* simpleExample.py, runAllExperiments.py, runAllAppendixResults.py or your own experiments based on this code at to the same level as _classes.py


## Running a standalone dataset

Now I will exemplify how to run the code with a standalone dataset. This example will use the baseball dataset which can be found in "data/numeric target/baseball.csv".

**Step 1**: is to import the dataset in the right format (pandas dataframe with the last column as the target variable) as:
```
# load data

import pandas as pd
datasetname= "baseball"
delim = ','
filename =  "./data/numeric target/"+datasetname+".csv"
df = pd.read_csv(filename,delimiter=delim)
```

**Step 2**: then after the dataset is initilized one can import the model and fit it to the data as in:
```
# initialize model and fit model to data
from _classes import SSDC

model = SSDC(task = "discovery")
model.fit(df)
```

**Step 3**: to analyse the results please refer to the attributes of SSDC in _classes.py, and as an example:
```
print(model) # returns the list of if-then-else if rules/subgroups with their respective antecedent description and consequent statistics
model.number_rules # returns the number of subgroups in the list
model.rule_sets # returns a list of the index of coverage of the subgroups description (with overlap between coverages)
model.measures # returns a dictionary of subgroup discovery measure results for the rule list and subgroups
```

**Step 4** (optional): in case you want to predict target values based on the model please provide a dataframe with the same format as the one used for model.fit and without the last column (target variable) as follows:
```
# example using the same data as used for fitting the model
datasetname= "baseball"
df = pd.read_csv(filename,delimiter=delim)
df2predict = df.iloc[:,:-1]
y , usageperrule = model.predict(df2predict)
```

*Pay attention to*:
* the code only accepts pandas dataframes.
* the pandas dataframe should contain the target variable in the last column.
* if an input variable is a number the algorithm considers it a numeric attribute.
* if an input variable is a string the algorithm  considers it a nominal/binary attribute (in case you have nominal attributes that are in intiger format please pass them to string).
* the algorithm preprocessing modifies the dataframe, in case it is needed for subsequent analysis we recommend to reload it again.

## Reproducing experiments in the paper

To fully reproduce the experiments presented in the paper run: 
```
runAllExperiments.py
```

and to run the experiments presented in the supplementary material run:
```
runAllAppendixResults.py
```


## References

A complete description of the theory behind this algorithm can be found in *extended_paper.pdf* 


These algorithm is also inspired in the previous work of:
* *M. Proença, Hugo* and *Van Leeuwen, Matthijs* :[Interpretable multiclass classification by MDL-based rule lists](https://arxiv.org/abs/1905.00328)


## Versioning

A new revamped version of this software is due very soon!

## Contact the author

* **Hugo Manuel Proença** - h.manuel.proenca@liacs.leidenuniv.nl

