#!/usr/bin/env python3

import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

DEBUG = True

def log(*msg):
    if DEBUG:
        s = ''
        for t in msg:
            s += str(t) + ' '
        sys.stdout.write('LOG: %s\n' % s)
        

bench_dir = os.path.join(os.getcwd(), 'bench')
for f in os.listdir(bench_dir):
    if 'png' in os.path.basename(f).split('.')[1]:
        os.unlink(os.path.join(bench_dir, f))

sns.set(style='ticks', color_codes=True)

calc_t = pd.read_csv(os.path.join(bench_dir, 'calc.csv'), sep = ';')
calc_g = sns.scatterplot(x='it', y='calc_t', data=calc_t)
calc_g.set(xscale='log')
calc_g.set(xlabel='Number of iterations', ylabel='Execution time (s)')
calc_g.set_title('Time spent vs iterations number')
calc_g.get_figure().savefig(os.path.join(bench_dir, 'exec_time.png'))
plt.clf()

test_dat = pd.read_csv(os.path.join(bench_dir, 'test_data.csv'))
test_g = sns.distplot(test_dat, kde=False, bins=10)
test_g.set_title('Test data')
test_g.get_figure().savefig(os.path.join(bench_dir,
                                'test_data.png'))

samples = pd.read_csv(os.path.join(bench_dir, 'samples.csv'), sep=';')
for it in samples['key'].unique():
    hist_dat = samples.loc[samples['key'] == it]
    hist_g = sns.distplot(hist_dat['val'], kde=False)
    hist_g.set_title('Boostrap data from test_data sample for %s iterations' % str(it))
    hist_g.get_figure().savefig(os.path.join(bench_dir,
                                'samples_hist-' + str(it) + '.png'))
    plt.clf()

for it in samples['key'].unique():
    hist_dat = samples.loc[samples['key'] == it]
    hist_g = sns.distplot(hist_dat['val'])
    hist_g.set_title('Boostrap data from test_data sample')
hist_g.get_figure().savefig(os.path.join(bench_dir, 'samples_hist-overlap.png'))
plt.clf()

samples_bxplt = sns.boxplot(y='val', x = 'key', data=samples)
samples_bxplt.set_title('Boostrap data from test samples')
samples_bxplt.set(xlabel='iterations')
samples_bxplt.get_figure().savefig(os.path.join(bench_dir, 'samples_bxplt-overlap.png'))
plt.clf()

fgrid_g = sns.FacetGrid(samples, col='key')
fgrid_g.map(sns.distplot, 'val', kde=False)
samples_bxplt.get_figure().savefig(os.path.join(bench_dir, 'samples_3hist.png'))
plt.clf()

params = pd.read_csv(os.path.join(bench_dir, 'params.csv'), sep=';')
params.columns = ['iteration_nb', 'param', 'val']

f0_dat = params.loc[params['param'] == 0]
f0_bplt = sns.boxplot(y='val', x = 'iteration_nb', data=f0_dat)
f0_bplt.set_title('f0 depending on iterations')
f0_bplt.set(xlabel='iterations')
f0_bplt.get_figure().savefig(os.path.join(bench_dir, 'f0_bplt.png'))
plt.clf()

a_dat = params.loc[params['param'] == 1]
a_bplt = sns.boxplot(y='val', x = 'iteration_nb', data=a_dat)
a_bplt.set_title('a depending on iterations')
a_bplt.set(xlabel='iterations')
a_bplt.get_figure().savefig(os.path.join(bench_dir, 'a_bplt.png'))
plt.clf()

