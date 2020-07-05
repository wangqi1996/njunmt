# coding=utf-8


def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def process(filename1, filename2):
    # filename1传汉语，filename2传另外一个语种
    l1_list = []
    l2_list = []
    raw = {}
    delete = 0

    with open(filename1 + ".new", 'w') as f1_w, open(filename2 + ".new", 'w') as f2_w:
        with open(filename1, 'r') as f1, open(filename2, 'r') as f2:
            for index, (l1, l2) in enumerate(zip(f1, f2)):
                # 重复句的删除
                h = str(hash(l1)) + str(hash(l2))
                if h in raw:
                    print(raw[h])
                    print(index)
                    delete += 1
                    continue
                raw[h] = index

                l1_list.append(l1.strip() + '\n')
                l2_list.append(l2.strip() + '\n')

                if len(l1_list) > 300:
                    f1_w.writelines(l1_list)
                    f2_w.writelines(l2_list)
                    l1_list = []
                    l2_list = []

        if len(l1_list) > 0:
            f1_w.writelines(l1_list)
            f2_w.writelines(l2_list)
    print("delete: ", delete)


if __name__ == '__main__':
    import sys

    print(sys.argv[1])
    print(sys.argv[2])
    process(sys.argv[1], sys.argv[2])
    # single_process(sys.argv[1])
