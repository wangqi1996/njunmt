# coding=utf-8

def add(d, key):
    d.setdefault(key, 0)
    d[key] += 1


def count_length(filename1):
    len1 = {}
    _len1 = 0
    count = 0
    with open(filename1, 'r') as f1:
        for l1 in f1:
            ll1 = len(l1.split(' '))
            _len1 += ll1
            if ll1 > 1000:
                print(l1)
            add(len1, int(ll1 / 10))
            count += 1
    print(len1)
    print(count)
    print(_len1)
    print(_len1 / count)


def cut_by_length(f1, len1=100):
    l1_list = []
    delete = 0
    with open(f1 + ".new", 'w') as f1_w:
        with open(f1, 'r') as f1:
            for l1 in f1:
                l1_len = len(l1.split(" "))

                if l1_len > len1:
                    delete += 1
                    continue

                l1_list.append(l1.strip('\n') + '\n')

                if len(l1_list) > 300:
                    f1_w.writelines(l1_list)
                    l1_list = []

            if len(l1_list) > 0:
                f1_w.writelines(l1_list)
    print(delete)


if __name__ == '__main__':
    import sys

    name = sys.argv[1]
    # name = "count"

    if name == "count":
        count_length(sys.argv[2])
    elif name == "cut":
        cut_by_length(sys.argv[2], int(sys.argv[3]))
