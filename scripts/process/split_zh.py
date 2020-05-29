# coding=utf-8

import re
import sys

import pkuseg


def split(filename):
    new_f = filename + '.new'
    c = []
    pattern = re.compile(r'[!?。\.\!\?]')
    with open(filename) as f1, open(new_f, 'w') as f2:
        for line in f1:
            m = re.findall(pattern, line)
            cc = re.split(pattern, line.strip('\n').strip())
            for i, t in enumerate(cc):
                if len(t) > 0 and len(m) > i:
                    c.append(t + m[i] + '\n')
                elif len(t) > 0:
                    print(t)
                    c.append(t + '\n')
            if len(c) > 300:
                f2.writelines(c)
                c = []
        if len(c) > 0:
            f2.writelines(c)


if __name__ == '__main__':
    pkuseg.test(sys.argv[1], sys.argv[2], nthread=20)
    # split("/home/user_data55/wangdq/code/mosesdecoder/scripts/ems/support/tt.txt")
