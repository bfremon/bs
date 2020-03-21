#!/usr/bin/env python3

def simple_dict2csv(d, fpath):
    '''write d dict to fpath csv file in pandas.DataFrame style.
    d must be composed of only simple lists
    '''
    write_buf = simple_dict2buf(d)
    f_w = open(fpath, 'w')
    f_w.write('%s' % write_buf)
    f_w.close()

    
def simple_dict2buf(d):
    col_len = len(d[list(d.keys())[0]])
    for k in d:
        _has_right_len(d[k], col_len)
    write_buf = ''
    for k in d:
        write_buf += str(k) + ';'
    write_buf += '\n'
    line_idx = 0
    while line_idx < col_len:
        s = ''
        for k in d:
            s += str(d[k][line_idx]) + ';'
        write_buf += s + '\n'
        line_idx += 1
    return write_buf
    
    
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
            _has_right_len(d[k][t], col_len)
        assert d[k].keys() == scd_keys
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
            simple_dict2csv(self.one_d, fp)
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
        
    unittest.main()
