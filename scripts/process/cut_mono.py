# coding=utf-8
import sys


def cut_mono(f_1, f_2):
    new_f1 = f_1 + '.new'
    new_f2 = f_2 + '.new'
    c1 = []
    c2 = []
    delete = 0
    with open(f_1) as f1, open(f_2) as f2:
        with open(new_f1, 'w') as nf1, open(new_f2, 'w') as nf2:
            for index, (l1, l2) in enumerate(zip(f1, f2)):
                if len(l1.strip().strip('\n').strip()) > 0 and len(l2.strip().strip('\n').strip()) > 0:
                    c1.append(l1)
                    c2.append(l2)
                    if len(c1) > 300:
                        nf1.writelines(c1)
                        nf2.writelines(c2)
                        c1 = []
                        c2 = []
                else:
                    delete += 1
                    print(index)
            if len(c1) > 0:
                nf1.writelines(c1)
                nf2.writelines(c2)
    print(delete)

if __name__ == '__main__':
    cut_mono(sys.argv[1], sys.argv[2])
