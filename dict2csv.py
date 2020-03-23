#!/usr/bin/env python3

DEBUG = True

def log(*msg):
    if DEBUG:
        import sys
        s = ''
        for t in msg:
            s += str(t) +  ' '
        sys.stdout.write('%s\n' % s)
        
            
def simple_dict2csv(d, fpath, header = True):
    '''write d dict to fpath csv file in pandas.DataFrame style.
    d must be composed of only simple lists.
    input: {'key1': [1,2,3], 'key2': [4, 5, 6] }
    output with header = True: 
    key1; key2;
    1;4; 
    2;5:
    3;6;
    '''
    write_buf = _simple_dict2buf(d, header)
    f_w = open(fpath, 'w')
    f_w.write('%s' % write_buf)
    f_w.close()

    
def _simple_dict2buf(d, header):
    col_len = len(d[list(d.keys())[0]])
    for k in d:
        _has_right_len(d[k], col_len)
    ret = ''
    if header:
        for k in d:
            ret += str(k) + ';'
        ret += '\n'
    line_idx = 0
    while line_idx < col_len:
        s = ''
        for k in d:
            s += str(d[k][line_idx]) + ';'
        ret += s + '\n'
        line_idx += 1
    return ret


def stack_simple_dict2csv(d, fpath, header = True):
    '''write d dict to fpath csv file in pandas.DataFrame style.
    d must be composed of only simple lists.
    input: {'key1': [1,2], 'key2': [4, 5, 6] }
    output with header = True : 
    key; val;
    key1; 1;
    key1; 2;
    key2; 4;
    ...
    '''
    write_buf = stack_simple_dict(d, header)
    f_w = open(fpath, 'w')
    f_w.write('%s' % write_buf)
    f_w.close()
    

def stack_simple_dict(d, header):
    for k in d:
        assert isinstance(d[k], list)
    ret = 'key; val;\n'
    if not header:
        ret = ''
    for k in d:
        for val in d[k]:
            ret += str(k) + ';' + str(val) + '\n';
    return ret
    
    
def _has_right_len(lst, lst_len): 
    if not isinstance(lst, list):
        raise SyntaxError('dict must be composed of only simple lists')
    if len(lst) != lst_len:
        raise SyntaxError('unbalanced dict')

    
def nested_dict2csv(d, fpath, label=None):
    '''write d dict to fpath csv file in pandas.DataFrame style.
    d must be composed of nested dict with the same number of keys,
    pointing to list of same len()
    '''
    fst_id = list(d.keys())[0]
    scd_id = list(d[fst_id].keys())[0]
    col_len = len(d[fst_id][scd_id])
    scd_keys = d[fst_id].keys()
    for k in d:
        for t in d[k]:
#            _has_right_len(d[k][t], col_len)
            assert d[k].keys() == scd_keys
    write_buf = '#'
    if label == None:
        write_buf = ';'
    else:
        write_buf += str(label) + ';'
    for k in d[fst_id]:
        write_buf += str(k) + ';'
    write_buf += '\n'
    line_idx = 0
    while line_idx < col_len:
        for k in d:
            s = ''
            s += str(k) + ';' 
            for t in d[k]:
                s += str(d[k][t][line_idx]) + ';'
            write_buf += s + '\n'
        line_idx += 1
    f_w = open(fpath, 'w')
    f_w.write('%s' % write_buf)
    f_w.close()
    

if __name__ == '__main__':
    import unittest
    import os
    
    class test_dict2csv(unittest.TestCase):
        one_d = { 'a': [1, 2, 3, 4],
                  'b': [5, 6, 7, 8],
                  'c': [9, 10, 11, 12] }
        two_d = { 'a': { '1': [1, 2, 3, 4, 5],
                         '2': [4, 5, 6, 5, 7],
                         '3': [2, 4, 5, 5, 6] },
                  'b': { '1': [7, 8, 9, 5, 8],
                         '2': [10, 11, 12, 6, 5],
                         '3': [23, 223, 33, 323, 33] }
        }

        
        def test_simple_dict2csv(self):
            fp = os.path.join(os.getcwd(), 'simple.csv')
            simple_dict2csv(self.one_d, fp, header = True)
            f_r = open(fp, 'r')
            line_idx = 0
            for l in f_r.readlines():
                if line_idx == 0:
                    self.assertTrue(l.split(';')[:-1] == ['a', 'b', 'c'])
                else:
                    cumsum = 0
                    for k in self.one_d:
                        cumsum += self.one_d[k][line_idx - 1]
                    csv_cumsum = 0
                    for t in l.split(';')[:-1]:
                        csv_cumsum += int(t)
                    self.assertTrue(cumsum == csv_cumsum)
                line_idx += 1
            f_r.close()
            os.unlink(fp)

            
        def test_stack_simple_dict2csv(self):
            fp = os.path.join(os.getcwd(), 'simple_stacked.csv')
            stack_simple_dict2csv(self.one_d, fp, header = False)
            f_r = open(fp ,'r')
            cumsum = {}
            for k in self.one_d:
                if not k in cumsum:
                    cumsum[k] = 0
                for val in self.one_d[k]:
                    cumsum[k] += val
            csv_cumsum = {}
            for l in f_r.readlines():
                k, v = l.split(';')[0], l.split(';')[1]
                if not k in csv_cumsum:
                    csv_cumsum[k] = 0
                csv_cumsum[k] += int(v)
            for k in cumsum:
                self.assertTrue(cumsum[k] == csv_cumsum[k])
            f_r.close()
            os.unlink(fp)
            
    unittest.main()
