#!/usr/bin/python3

import os
import sys
import pandas as pd
import weibull

#iter_nb = 100000
iter_nb = 100

if len(sys.argv) != 2:
    print("input file in CSV format (; as separator) needed (column 1 labeled 'cat', column 2 labeled 'val')")
    sys.exit(-1)
in_f = os.path.join(sys.argv[1])
print(in_f)
in_dat = pd.read_csv(in_f, sep=';')
fit_dat = weibull.batch_fit_wb_cdf(in_dat, os.path.join(os.getcwd(), 'fit_dat.csv'), iter_nb)
sts = weibull.stats_fit_wb_cdf(fit_dat)
sts.to_csv('stats.csv', sep=';')
print(sts)
weibull.plt_fit_wb_cdf(in_dat, fit_dat, os.getcwd())
