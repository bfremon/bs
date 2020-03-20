#!/usr/bin/env python3

import sys
import numpy as np
import statsmodels.api as sm

DEBUG = True

def log(*msg):
    if DEBUG:
        s = ''
        for t in msg:
            s += str(t) + ' '
        sys.stdout.write('%s\n' % s)


def resample(v, resample_n=10):
    '''
    return resample_n samples taken randomly from 1D scalar v
    '''
    if resample_n > len(v): 
        raise SyntaxError('vector lenght should be inferior to resample size')
    i = 0
    ret = []
    while i < resample_n:
        idx = int(len(v) * (np.random.uniform()))
        ret.append(v[idx])
        i += 1
    return ret


def bootstrap(seed, func, resample_n = 10, iter_n = 1000):
    '''
    Randomly sample iter_n times resample_n samples from vec ins bs_arr,
    collect the output of func(bs_arr) and the boostrap samples used.
    '''
    i = 0
    ret = {}
    while i < iter_n:
        bs_arr = resample(seed, resample_n)
        ret[i] = {}
        ret[i]['func_ret'] = func(bs_arr)
        ret[i]['samples'] = bs_arr
        i += 1
    return ret


def split_bs_out(dat):
    '''
    Split output from boostrap() into samples and params (from 'func_ret'). 
    '''
    assert 'func_ret' in dat[0]
    assert 'samples' in dat[0]
    assert len(dat[0]['func_ret']) >= 1
    parms = {}
    for k in range(0, len(dat[0]['func_ret'])):
        parms[k] = []
    smpls_lst = []
    for i in dat:
        for j in dat[i]:
            if j == 'samples':
                smpls_lst.append(dat[i][j])
            elif j == 'func_ret':
                for k in range(0, len(dat[i][j])):
                    parms[k].append(dat[i][j][k])
            else:
                raise KeyError('split_bs_out(): %s is not a valid key'
                               % str(j))
    smpls = []
    for sublist in smpls_lst:
        for item in sublist:
            smpls.append(item)
    return parms, smpls


if __name__ == '__main__':
    import unittest

    class test_bootstrap(unittest.TestCase):

        data = [ i for i in range(0, 100) ]
        
        def _return_arr(self, a):
                return a

            
        def test_resample(self):
            r = resample(self.data, len(self.data))
            self.assertTrue(len(r) == len(self.data))
            for e in r:
                assert e in self.data
            self.assertRaises(SyntaxError, resample, (self.data, 100))
            

        def test_bootstrap(self):
            iter_n = 100
            resample_n = 50
            r = bootstrap(self.data, self._return_arr, resample_n = resample_n, iter_n = iter_n)
            func_ret = []
            samples = []
            for k in r:
                for t in r[k]['func_ret']:
                    func_ret.append(t)
                for t in r[k]['samples']:
                    samples.append(t)
            self.assertTrue(len(r) == iter_n)
            self.assertTrue(len(func_ret) == iter_n * resample_n)
            self.assertTrue(len(samples) == iter_n * resample_n)
            for i in range(0, iter_n):
                for j in range(0, iter_n):
                    if j != i:
                        self.assertTrue(r[i]['samples'] != r[j]['samples'])
                        
            
        def test_split_bs_out(self):
            iter_n = 100
            resample_n = 50
            dat = bootstrap(self.data, self._return_arr, resample_n = resample_n, iter_n = iter_n)
            params, samples = split_bs_out(dat)
            self.assertTrue(len(samples) == iter_n * resample_n)
            self.assertTrue(len(params) == resample_n)
            for p in params:
                self.assertTrue(len(params[p]) == iter_n)
            
    unittest.main()
