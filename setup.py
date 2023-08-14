#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
import numpy
import os

os.environ['CC'] = 'g++';

accelerated_sequence_clustering_module = Extension('_accelerated_sequence_clustering', language = "c++",
                           sources=['accelerated_sequence_clustering_wrap.cxx', 'accelerated_sequence_clustering.cpp'],
                           include_dirs = [numpy.get_include(), '.'],
                           extra_compile_args=''
                           )

setup (name = 'accelerated_sequence_clustering',
       version = '1.0.1',
       author      = "Reza Mortazavi",
       description = """accelerated clustering of sequential data""",
       ext_modules = [accelerated_sequence_clustering_module],
       py_modules = ["accelerated_sequence_clustering"],
       zip_safe=False
       )