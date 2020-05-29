# coding=utf-8

def add(d, key):
    d.setdefault(key, 0)
    d[key] += 1


def count_length(filename1, filename2):
    len1 = {}
    _len1 = 0
    len2 = {}
    _len2 = 0
    count = 0
    l1l2 = {}
    l2l1 = {}
    with open(filename1, 'r') as f1, open(filename2, 'r') as f2:
        for l1, l2 in zip(f1, f2):
            ll1 = len(l1.split(' '))
            ll2 = len(l2.split(' '))
            _len1 += ll1
            _len2 += ll2
            add(len1, int(ll1 / 10))
            add(len2, int(ll2 / 10))
            add(l1l2, int(ll1 / ll2))
            add(l2l1, int(ll2 / ll1))
            if int(ll2 / ll1) > 10:
                print(l1)
                print(l2)
            count += 1
    print(len1)
    print(len2)
    print(count)
    print(_len1)
    print(_len2)
    print(_len1 / count)
    print(_len2 / count)
    print(l1l2)
    print(l2l1)


def cut_by_length(f1, f2, len1=100, len2=100, l1l2=3, l2l1=6):
    l1_list = []
    l2_list = []
    delete = 0
    delete2 = 0
    with open(f1 + ".new", 'w') as f1_w, open(f2 + ".new", 'w') as f2_w:
        with open(f1, 'r') as f1, open(f2, 'r') as f2:
            for l1, l2 in zip(f1, f2):
                l1_len = len(l1.split(" "))
                l2_len = len(l2.split(" "))

                if l1_len > len1 or l2_len > len2:
                    delete += 1
                    continue
                if int(l1_len / l2_len) > l1l2 or int(l2_len / l1_len) > l2l1:
                    delete2 += 1
                    continue

                l1_list.append(l1.strip('\n') + '\n')
                l2_list.append(l2.strip('\n') + '\n')

                if len(l1_list) > 300:
                    f1_w.writelines(l1_list)
                    f2_w.writelines(l2_list)
                    l1_list = []
                    l2_list = []

            if len(l1_list) > 0:
                f1_w.writelines(l1_list)
                f2_w.writelines(l2_list)
    print(delete)
    print(delete2)


if __name__ == '__main__':
    import sys

    name = sys.argv[1]
    # name = "count"

    if name == "count":
        count_length(sys.argv[2], sys.argv[3])
        # count_length("/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.uy", "/home/user_data55/wangdq/data/ccmt/uy-zh/test/test.txt")
    elif name == "cut":
        cut_by_length(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
