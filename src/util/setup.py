# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:00:36 2020

@author: gathu
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("fastmean_var.pyx",
                            compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()]
)