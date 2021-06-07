# -*- coding: utf-8 -*-
# @Time : 2021/2/15 20:59
# @Author : luff543
# @Email : luff543@gmail.com
# @File : io_functions.py
# @Software: PyCharm

import csv
import pandas as pd


def read_csv(file_name, names, delimiter='t'):
    if delimiter == 't':
        sep = '\t'
    elif delimiter == 'b':
        sep = ' '
    else:
        sep = delimiter
    # return pd.read_csv(file_name, sep=sep, quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=names, keep_default_na=False, na_filter = True, na_values=['<DEFINE_NULL>'])
    # return pd.read_csv(file_name, sep=sep, quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=names,
    #                    keep_default_na=True, na_filter=True, na_values=['<DEFINE_NULL>'])
    return pd.read_csv(file_name, sep=sep, quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=names,
                       keep_default_na=False)
    # return pd.read_csv(file_name, sep=sep, quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=names)
