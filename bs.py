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
    i = 0
    ret = {}
    while i < iter_n:
        bs_arr = resample(seed, resample_n)
        ret[i] = {}
        ret[i]['func_ret'] = func(bs_arr)
        ret[i]['samples'] = bs_arr
        i += 1
    return ret


if __name__ == '__main__':
    
    import unittest
    class test_bootstrap(unittest.TestCase):

        def test_resample(self):
            v  = [ i for i in range(0, 100) ]
            ret = resample(v, len(v))
            self.assertTrue(len(ret) == len(v))
            for e in ret:
                assert e in v
            self.assertRaises(SyntaxError, resample, (v, 100))
            

        def test_bootstrap(self):
            def return_arr(a):
                return a
            a = [i for i in range(0, 100)]
            iter_n = 100
            resample_n = 50
            ret = bootstrap(a, return_arr, resample_n = resample_n, iter_n = iter_n)
            func_ret = []
            samples = []
            for k in ret:
                for t in ret[k]['func_ret']:
                    func_ret.append(t)
                for t in ret[k]['samples']:
                    samples.append(t)
            self.assertTrue(len(ret) == iter_n)
            self.assertTrue(len(func_ret) == iter_n * resample_n)
            self.assertTrue(len(samples) == iter_n * resample_n)

            
    unittest.main()
