# -*- coding: utf-8 -*-

import pandas as pd
file = "C:/Users/gathu/surfdrive/PhD/MDL/code/MDL subgroup scikit - development/data/numeric target/originals/bikesharing_hour.csv"
df = pd.read_csv(file)
df = df.drop(["instant","dteday","casual","registered"],axis =1)

def test_func(df):
    """ Test Function for generating new value"""
    if df['season'] == 1:
        return "winter"
    elif df['season'] == 2:
        return "spring"
    elif df['season'] == 3:
        return "summer"
    elif df['season'] == 4:
        return "autumn"

def holiday(df):
    """ Test Function for generating new value"""
    if df['holiday'] == 0:
        return "no"
    elif df['holiday'] == 1:
        return "yes"
df["holiday"] = df.apply(holiday, axis=1)

def workingday(df):
    """ Test Function for generating new value"""
    if df['workingday'] == 0:
        return "no"
    elif df['workingday'] == 1:
        return "yes"

df["workingday"] = df.apply(workingday, axis=1)
file = "C:/Users/gathu/surfdrive/PhD/MDL/code/MDL subgroup scikit - development/data/numeric target/originals/bikesharing_hourchanged.csv"
df.to_csv(file, index=False)

import pandas as pd
file = "C:/Users/gathu/surfdrive/PhD/MDL/code/MDL subgroup scikit - development/data/numeric target/originals/cholesterol.csv"
df = pd.read_csv(file)
df.drop(df[df.thal == "?"].index, inplace=True)
df.thal = pd.to_numeric(df.thal)
df.drop(df[df.ca == "?"].index, inplace=True)
df.ca = pd.to_numeric(df.ca)

df=df.dropna()
def sex(df):
    """ Test Function for generating new value"""
    if df['sex'] == 1:
        return "male"
    elif df['sex'] == 0:
        return "female"    
df["sex"] = df.apply(sex, axis=1)

def cp(df):
    """ Test Function for generating new value"""
    if df['cp'] == 1:
        return "typical angina"
    elif df['cp'] == 2:
        return "atypical angina"
    elif df['cp'] == 3:
        return "non-anginal pain"
    elif df['cp'] == 4:
        return "asymptomatic"
df["cp"] = df.apply(cp, axis=1)

def fbs(df):
    """ Test Function for generating new value"""
    if df['fbs'] == 1:
        return "yes"
    elif df['fbs'] == 0:
        return "no"
df["fbs"] = df.apply(fbs, axis=1)

def restecg(df):
    """ Test Function for generating new value"""
    if df['restecg'] == 0:
        return "normal"
    elif df['restecg'] == 1:
        return "ST-T wave abnormality"
    elif df['restecg'] == 2:
        return "left ventricular hypertrophy"
df["restecg"] = df.apply(restecg, axis=1)

def exang(df):
    """ Test Function for generating new value"""
    if df['exang'] == 1:
        return "yes"
    elif df['exang'] == 0:
        return "no"
df["exang"] = df.apply(exang, axis=1)

def slope(df):
    """ Test Function for generating new value"""
    if df['slope'] == 1:
        return "upsloping"
    elif df['slope'] == 2:
        return "flat"
    elif df['slope'] == 3:
        return "downslopping"
df["slope"] = df.apply(slope, axis=1)
    
def thal(df):
    """ Test Function for generating new value"""
    if df['thal'] == 3:
        return "normal"
    elif df['thal'] == 6:
        return "fixed defect"
    elif df['thal'] == 7:
        return "reversal defect"    
df["thal"] = df.apply(thal, axis=1)

file = "C:/Users/gathu/surfdrive/PhD/MDL/code/MDL subgroup scikit - development/data/numeric target/originals/cholesterolchange.csv"
df.to_csv(file, index=False)





import pandas as pd
file = "C:/Users/gathu/surfdrive/PhD/MDL/code/MDL subgroup scikit - development/data/numeric target/originals/moneyball.csv"
df = pd.read_csv(file)
df.drop(df[df.thal == "?"].index, inplace=True)
df.thal = pd.to_numeric(df.thal)
df.drop(df[df.ca == "?"].index, inplace=True)
df.ca = pd.to_numeric(df.ca)

df=df.dropna()
def sex(df):
    """ Test Function for generating new value"""
    if df['sex'] == 1:
        return "male"
    elif df['sex'] == 0:
        return "female"    
df["sex"] = df.apply(sex, axis=1)

def cp(df):
    """ Test Function for generating new value"""
    if df['cp'] == 1:
        return "typical angina"
    elif df['cp'] == 2:
        return "atypical angina"
    elif df['cp'] == 3:
        return "non-anginal pain"
    elif df['cp'] == 4:
        return "asymptomatic"
df["cp"] = df.apply(cp, axis=1)

def fbs(df):
    """ Test Function for generating new value"""
    if df['fbs'] == 1:
        return "yes"
    elif df['fbs'] == 0:
        return "no"
df["fbs"] = df.apply(fbs, axis=1)

def restecg(df):
    """ Test Function for generating new value"""
    if df['restecg'] == 0:
        return "normal"
    elif df['restecg'] == 1:
        return "ST-T wave abnormality"
    elif df['restecg'] == 2:
        return "left ventricular hypertrophy"
df["restecg"] = df.apply(restecg, axis=1)

def exang(df):
    """ Test Function for generating new value"""
    if df['exang'] == 1:
        return "yes"
    elif df['exang'] == 0:
        return "no"
df["exang"] = df.apply(exang, axis=1)

def slope(df):
    """ Test Function for generating new value"""
    if df['slope'] == 1:
        return "upsloping"
    elif df['slope'] == 2:
        return "flat"
    elif df['slope'] == 3:
        return "downslopping"
df["slope"] = df.apply(slope, axis=1)
    
def thal(df):
    """ Test Function for generating new value"""
    if df['thal'] == 3:
        return "normal"
    elif df['thal'] == 6:
        return "fixed defect"
    elif df['thal'] == 7:
        return "reversal defect"    
df["thal"] = df.apply(thal, axis=1)

file = "C:/Users/gathu/surfdrive/PhD/MDL/code/MDL subgroup scikit - development/data/numeric target/originals/cholesterolchange.csv"
df.to_csv(file, index=False)
