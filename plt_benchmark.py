#!/usr/bin/env python3

import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

bench_dir = os.path.join(os.getcwd(), 'bench')
calc_t = pd.read_csv(os.path.join(bench_dir, 'calc.csv'), sep = ';')
calc_ax = sns.scatterplot(x='#it', y='calc_t', data=calc_t)
calc_ax.set(xscale='log')
calc_ax.set(xlabel='Number of iterations', ylabel='Execution time (s)')
calc_ax.get_figure().savefig(os.path.join(bench_dir, 'exec_time.png'))
