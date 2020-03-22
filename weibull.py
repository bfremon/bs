#!/usr/bin/env python3

import numpy as np
import statsmodels.api as sm
import pandas as pd
import sys
import bs
import serialize_dict

DEBUG = True

def log(*msg):
    if DEBUG:
        s = ''
        for t in msg:
            s += str(t) + ' '
        sys.stdout.write('%s\n' % s)


def fit_weibull_cdf(v):
    '''
    fit data from v array with weibull CDF given by:
    p(x) = 1 - exp((-x / x0)^(-a))
    return x0, a
    '''
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
    return x0, alpha


def pre_process(filepath):
    '''
    process a CSV file with:
    column 0: categorical data
    column 1: variable data
    return a Pandas.DataFrame
    '''
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


def get_ci(v, alpha = 0.05):
    assert 0 <= alpha <= 1
    ''' return 1 - alpha / 2 confidence intervales for v'''
    lower_bound = alpha / 2
    upper_bound = 1 - lower_bound
    ser = pandas.Series(v)
    return (ser.quantile(lower_bound), ser.quantile(upper_bound))


def bench_ci(v, iters):
    '''benchmark fit_weibull_cdf() by trying iterations numbers in iter_list.
    return stats about parameters and execution time.
    '''
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
    serialize_dict.simple_dict2csv(calc_t, os.path.join(dest_dir, 'calc.csv'))
    serialize_dict.nested_dict2csv(prms, os.path.join(dest_dir, 'params.csv'))
    serialize_dict.stack_simple_dict2csv(smpls, os.path.join(dest_dir, 'samples.csv'))


if __name__ == '__main__':
    data = [ 1480, 1558, 1661, 1705, 1753, 1824, 1833,
             1901, 2125, 2189, 2226, 2261, 2377, 2433,
             2468, 2608, 2638, 2646, 2650, 2697, 2716,
             2899, 2959, 3017, 3044, 3051, 3146, 3220, 3299 ] 

    if len(sys.argv) == 2 and sys.argv[1] == '-b':
        bench_ci(data, [10**i for i in range(3, 6)])
    else:
        import unittest
        import os
        
        class test_weibull(unittest.TestCase):
            data = data
            def test_fit_weibull_cdf(self):
                x0, a = fit_weibull_cdf(self.data)
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
        
        unittest.main()
