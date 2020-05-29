# coding=utf-8


def cut_by_align(f1_name, f2_name, forward, backward):
    newf1 = f1_name + '.new'
    newf2 = f2_name + '.new'
    c1 = []
    c2 = []
    delete = 0
    with open(f1_name, 'r') as f1, open(f2_name, 'r') as f2, open(forward, 'r') as fo, open(backward, 'r') as ba:
        with open(newf1, 'w') as nf1, open(newf2, 'w') as nf2:
            for index, (l1, l2, o, b) in enumerate(zip(f1, f2, fo, ba)):
                len_f1 = len(l1.split(' '))
                len_f2 = len(l2.split(' '))
                len_o = len(o.split(' '))
                len_b = len(b.split(' '))
                if len_o > len_f1 * 0.55 or len_b > len_f2 * 0.55:
                    c1.append(l1)
                    c2.append(l2)
                else:
                    print(index)
                    delete += 1
                if len(c1) > 300:
                    nf1.writelines(c1)
                    nf2.writelines(c2)
                    c1 = []
                    c2 = []
            if len(c1) > 0:
                nf1.writelines(c1)
                nf2.writelines(c2)
    print("delete", delete)


if __name__ == '__main__':
    import sys

    f1, f2, f3, f4 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    # cut_by_align(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    # f1 = "/home/user_data55/wangdq/data/ccmt/zh-en/parallel/zh.token"
    # f2 = "/home/user_data55/wangdq/data/ccmt/zh-en/parallel/en.token"
    # f3 = "/home/user_data55/wangdq/data/ccmt/zh-en/parallel/zh_en_align"
    # f4 = "/home/user_data55/wangdq/data/ccmt/zh-en/parallel/en_zh_align"

    cut_by_align(f1, f2, f3, f4)
