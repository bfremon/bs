#!/usr/bin/env python3

DEBUG = False

def log(*msg):
    if DEBUG:
        import sys
        s = ''
        for t in msg:
            s += str(t) +  ' '
        sys.stdout.write('LOG: %s\n' % s)
        
            
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
        ret = ret[:-1] + '\n'
    line_idx = 0
    while line_idx < col_len:
        s = ''
        for k in d:
            s += str(d[k][line_idx]) + ';'
        ret += s
        ret = ret[:-1] + '\n'
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
    ret = 'key;val\n'
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
    pointing to lists of same len()
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
    write_buf = write_buf[:-1] + '\n'
    line_idx = 0
    while line_idx < col_len:
        for k in d:
            s = ''
            s += str(k) + ';' 
            for t in d[k]:
                s += str(d[k][t][line_idx]) + ';'
            write_buf += s 
            write_buf = write_buf[:-1] + '\n'
        line_idx += 1
    f_w = open(fpath, 'w')
    f_w.write('%s' % write_buf)
    f_w.close()

    
def melt_nested_dict2csv(d, fpath):
    '''
    Transform d (nested dict with two levels in long format.
    List at 3rd level can be of arbitrary lenght
    d = {'a': { 1 : [], 2: [], ...}, 'b': {'a': [], 'b': []}
    transformed to :
    a; 1; a[1][0],
    a; 1; a[1][1],
    ...
    '''
    s = ''
    line_cnt = 0 
    for k_1 in d:
        for k_2 in d[k_1]:
            for i in range(0, len(d[k_1][k_2])):
                s += str(k_1) + ';' + str(k_2) + ';' \
                           + str(d[k_1][k_2][i]) + '\n'
                line_cnt += 1
    log('melt_nested_dict2csv() - line count %i' % line_cnt)
    f_w = open(fpath, 'w')
    f_w.write('%s' % s)
    f_w.close()

                           
def lst2csv(lst, fpath):
    f_w = open(fpath, 'w')
    s = ''
    for t in lst:
        s += str(t) + '\n'
    f_w.write('%s' % s)
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
        three_d = { 'a': { '1': [1, 2, 3, 4, 5],
                         '2': [4, 5, 6, 5, 7],
                         '3': [2, 4, 5, 5, 6] },
                    'b': { '1': [7, 8, 9, 5, 8],
                           '2': [10, 11, 12, 6, 5, 5, 44],
                           '3': [23, 223, 33, 323, 33] },
                    'c': { '1': [7, 8, 9, 5, 8],
                           'd': [10, 11, 12, 6, 5, 5, 44],
                           '3': [23, 223, 33, 323, 33], 
                           '4': [2, 4, 5, 5, 6] }
                    }

        lst = [1, 2, 3, 4, 5]
        
        def test_simple_dict2csv(self):
            fp = os.path.join(os.getcwd(), 'simple.csv')
            simple_dict2csv(self.one_d, fp, header = True)
            f_r = open(fp, 'r')
            line_idx = 0
            for l in f_r.readlines():
                if line_idx == 0:
                    keys_lst = l.replace('\n', '')
                    keys_lst = keys_lst.split(';')
                    self.assertTrue(keys_lst == ['a', 'b', 'c'])
                else:
                    cumsum = 0
                    for k in self.one_d:
                        cumsum += self.one_d[k][line_idx - 1]
                    csv_cumsum = 0
                    for t in l.split(';'):
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


        def test_lst2csv(self):
            fp = os.path.join(os.getcwd(), 'lst.csv')
            lst2csv(self.lst, fp)
            cumsum = 0
            for i in self.lst:
                cumsum += i
            f_r = open(fp, 'r')
            csv_cumsum = 0
            for l in f_r.readlines():
                csv_cumsum += int(l.replace('\n', ''))
            self.assertTrue(csv_cumsum == cumsum)
            f_r.close()
            os.unlink(fp)


        def test_nested_dict2csv(self):
            fp = os.path.join(os.getcwd(), 'nested.csv')
            nested_dict2csv(self.two_d, fp)
            f_r = open(fp, 'r')
            cumsum = {}
            for i in self.two_d:
                for j in self.two_d[i]:
                    if not j in cumsum:
                        cumsum[j] = 0
                    for k in range(0, len(self.two_d[i][j])):
                        cumsum[j] += int(self.two_d[i][j][k])
            csv_cumsum = {}
            csv_cumsum[1], csv_cumsum[2], csv_cumsum[3] = 0, 0, 0
            line_idx = 0
            for l in f_r.readlines():
                if line_idx == 0:
                    pass
                else:
                    s = l.split(';')[1:]
                    csv_cumsum[1] += int(s[0])
                    csv_cumsum[2] += int(s[1])
                    csv_cumsum[3] += int(s[2])
                line_idx += 1
            for t in csv_cumsum:
                self.assertTrue(csv_cumsum[t] == cumsum[str(t)])
            f_r.close()
            os.unlink(fp)


        def test_melt_nested_dict2csv(self):
            fp = os.path.join(os.getcwd(), 'melt_nested.csv')
            melt_nested_dict2csv(self.three_d, fp)
            cumsum = {}
            for k_1 in self.three_d:
                if not k_1 in cumsum:
                    cumsum[k_1] = {}
                for k_2 in self.three_d[k_1]:
                    if not k_2 in cumsum:
                        cumsum[k_1][k_2] = 0
                    for i in range(0, len(self.three_d[k_1][k_2])):
                        cumsum[k_1][k_2] += int(self.three_d[k_1][k_2][i])
            cum_len = 0
            for k in self.three_d:
                for j in self.three_d[k]:
                    cum_len += len(self.three_d[k][j])
            csv_cumsum = {}
            f_r = open(fp, 'r')
            line_cnt = 0
            for l in f_r.readlines():
                s = l.split(';')
                k_1 = s[0]
                k_2 = s[1]
                val = s[2]
                if not k_1 in csv_cumsum:
                    csv_cumsum[k_1] = {}
                if not k_2 in csv_cumsum[k_1]:
                    csv_cumsum[k_1][k_2] = 0
                csv_cumsum[k_1][k_2] += int(val)
                line_cnt += 1
            self.assertTrue(cumsum == csv_cumsum)
            self.assertTrue(cum_len == line_cnt)
            f_r.close()
            os.unlink(fp)

        
    unittest.main()
