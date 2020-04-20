#!/usr/bin/env python3

import numpy as np
import statsmodels.api as sm
import pandas as pd
import sys
import os
import bs
import struct2csv
import seaborn as sns
import matplotlib.pyplot as plt

DEBUG = True

def log(*msg):
    if DEBUG:
        s = ''
        for t in msg:
            s += str(t) + ' '
        sys.stdout.write('%s\n' % s)


def fit_wb_cdf(v):
    """
    fit data from v array with weibull CDF given by:
    p(x) = 1 - exp((-x / x0)^(-a))
    return x0, a
    """
    vec = np.array(sorted(v)).astype('float')
    cdf = sm.distributions.empirical_distribution.ECDF(vec)
    # min (= 0) and max(= 1.0) quantile removed to allow log transform
    p = np.delete(cdf.y, 0)
    p = np.delete(p, -1)
    x = np.delete(cdf.x, 0)
    x = np.delete(x, -1)
    y = np.log(- np.log(1 - p))
    ln_x = np.log(x)
    x_cte = sm.add_constant(ln_x)
    model = sm.OLS(y, x_cte).fit()
    preds = model.predict(x_cte)
    # y = beta + alpha * x
    beta, alpha = model.params
    x0 = np.exp(-beta / alpha)
    # TODO: param to control number of decimals places
    return x0, alpha


def pre_process(filepath):
    """
    process a CSV file with:
    column 0: categorical data
    column 1: variable data
    return a Pandas.DataFrame
    """
    ret = {}
    f_r = open(filepath, 'r')
    first_line = True
    current_category = ''
    for l in f_r.readlines():
        try:
            cat, data = l.split(';')[0], l.split(';')[1]
        except IndexError:
            raise IndexError('malformed input')
        if l[0] == '#':
            next
        elif first_line and current_category == '':
            ret[cat] = []
            ret[cat].append(data)
            current_category = cat
            first_line = False
        elif cat != current_category:
            ret[cat] = []
            current_category = cat
            ret[cat].append(data)
        elif cat == current_category:
            ret[cat].append(data)
    f_r.close()
    return pd.DataFrame(ret)


def bench_ci(v, iters):
    """
    Benchmark fit_weibull_cdf() by trying iterations numbers in iter_list.
    return stats about parameters and execution time.
    """
    import time
    import os

    dest_dir = os.path.join(os.getcwd(), 'bench')
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    else:
        for f in os.listdir(dest_dir):
            try:
                os.unlink(os.path.join(dest_dir, f))
            except:
                raise IOError('Impossible to remove %s\n' % f)
            
    calc_t = {'it': [], 'calc_t': []}
    prms = {}
    smpls = {}
                          
    for it in iters:
        sys.stdout.write('Processing %i iterations:' % it)
        start_t = time.time()
        out = bs.bootstrap(v, fit_weibull_cdf, resample_n = len(v), iter_n = it)
        params, samples = bs.split_bs_out(out)
        prms[it] = params
        smpls[it] = samples
        end_t = time.time() - start_t
        calc_t['it'].append(it)
        calc_t['calc_t'].append(end_t)
        sys.stdout.write(' %3.1f s\n' % end_t) 
    struct2csv.simple_dict2csv(calc_t, os.path.join(dest_dir, 'calc.csv'))
    struct2csv.melt_nested_dict2csv(prms, os.path.join(dest_dir, 'params.csv'))
    struct2csv.stack_simple_dict2csv(smpls, os.path.join(dest_dir, 'samples.csv'))

    
def batch_fit_wb_cdf(input_data, out_f_p, iter_nb=10000):
    """
    Fit a weibull cdf on each category of input_data for iter_nb iterations.
    Return a melted dataframe with all data, logged in out_f_p (csv file).
    """
    r = {}
    out_f_p = os.path.join(out_f_p)
    for cat in input_data['cat'].unique():
        log('Processing category %s' % str(cat))
        dat = input_data[input_data['cat'] == cat]
        v = dat['val'].tolist()
        out = bs.bootstrap(v, fit_wb_cdf, resample_n = len(v), iter_n = iter_nb)
        params, samples = bs.split_bs_out(out)
        r[cat] = params
    struct2csv.melt_nested_dict2csv(r, out_f_p)
    ret = pd.read_csv(out_f_p, header=None, sep=';')
    ret.columns = ['cat', 'param', 'val']
    return ret


def stats_fit_wb_cdf(df, alpha=0.05):
    """
    Return a dataframe with median and 1 - alpha % CI for weibull cdf fits 
    in df (output of batch_fit_wb_cdf) 
    """
    if not 0 < alpha < 1:
        raise SyntaxError('alpha must be strictly between 0 and 1')
    r = {}
    for cat in df['cat'].unique():
        cat_dat = df[df['cat'] == cat]
        a_dat = cat_dat[cat_dat['param'] == 1]
        f0_dat = cat_dat[cat_dat['param'] == 0]
        a_median = a_dat['val'].quantile(0.5)
        lb = alpha / 2
        ub = 1 - lb
        a_lb = a_dat['val'].quantile(lb)
        a_ub = a_dat['val'].quantile(ub)
        f0_median = f0_dat['val'].quantile(0.5)
        f0_lb = f0_dat['val'].quantile(lb)
        f0_ub = f0_dat['val'].quantile(ub)
        r[cat] = [a_lb, a_median, a_ub, f0_lb, f0_median, f0_ub]
    index = ['a lower bound', 'a median', 'a upper bound',
             'f0 lower bound', 'f0 median', 'f0 upper bound']
    ret = pd.DataFrame(r, index=index)
    return ret


def plt_fit_wb_cdf(in_df, out_df, out_d_p):
    """
    Plot different graphs from in_df (raw bootstrap data dataframe) 
    and out_df (output of batch_fit_wb_cdf())
    """
    dat_out_f = os.path.join(out_d_p, 'input.png')
    _bxplt('cat', 'val', in_df,  dat_out_f, title='Input data')
    a = out_df[out_df['param'] == 1]
    f0 = out_df[out_df['param'] == 0]
    a_out_f = os.path.join(out_d_p, 'bootstrap_a.png')
    _bxplt('cat', 'val', a,  a_out_f, title='Bootstrap a (10^5 iterations)')
    f0_out_f = os.path.join(out_d_p, 'bootstrap_f0.png')
    _bxplt('cat', 'val', f0, f0_out_f, title='Bootstrap f0 (10^5 iterations)')
    
def _bxplt(x, y, dat, out_f, title=None, ytitle=None):
    bxp = sns.boxplot(y=y, x=x, data=dat)
    if title != None:
        bxp.set_title(title)
        bxp.set_xticklabels(bxp.get_xticklabels(), rotation=30)
    if ytitle != None:
        raise ErrorNotImplemented
    bxp.get_figure().savefig(out_f)
    plt.clf()


if __name__ == '__main__':
    data = [ 1480, 1558, 1661, 1705, 1753, 1824, 1833,
             1901, 2125, 2189, 2226, 2261, 2377, 2433,
             2468, 2608, 2638, 2646, 2650, 2697, 2716,
             2899, 2959, 3017, 3044, 3051, 3146, 3220, 3299 ] 

    if len(sys.argv) == 2 and sys.argv[1] == '-b':
        bench_ci(data, [ 10 ** i for i in range(3, 6)])
        struct2csv.lst2csv(data, os.path.join(os.getcwd(), 'bench', 'test_data.csv'))
    else:
        import unittest
        import os
        
        class test_weibull(unittest.TestCase):
            data = data
            batch_d = {'a': data, 'b': data, 'c': data}
            
            def test_fit_wb_cdf(self):
                x0, a = fit_wb_cdf(self.data)
                self.assertAlmostEqual(x0, 2613, delta = 10)
                self.assertAlmostEqual(a, 5, delta = 0.5)
            
            def test_preprocess(self):
                fpath = os.path.join(os.getcwd(), 'test.csv')
                f_w = open(fpath, 'w')
                cat_nb = 5
                f_w.write('%s%s' % ('#cat;data;', os.linesep))
                for cat in range(0, cat_nb):
                    for d in self.data:
                        f_w.write('%s;%s;%s' % (str(cat), str(d), os.linesep))
                f_w.close()
                ret = pre_process(fpath)
                self.assertTrue(len(ret.columns) == cat_nb)
                for k in ret.columns:
                    assert len(ret[k]) == len(self.data)
                os.unlink(fpath)

            def test_batch_wb(self):
                df = pd.DataFrame(self.batch_d).melt()
                df.columns = ['cat', 'val']
                out_f_p = os.path.join(os.getcwd(), 'batch-test.csv')
                res = batch_fit_wb_cdf(df, out_f_p, 100)
                self.assertTrue(len(res) == 600)
                sts = stats_fit_wb_cdf(res)
#                print(sts)
                os.unlink(out_f_p)

            def test_stats_fit_wb_cdf(self):
                df = pd.DataFrame(self.batch_d)
                self.assertRaises(SyntaxError, stats_fit_wb_cdf, df, 2)
        

        unittest.main()
