# coding=utf-8

import re


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


def single_process(filename1):
    # filename1传汉语，filename2传另外一个语种
    l1_list = []
    raw = {}
    delete = 0
    # wmt抄的
    re_space = re.compile(r"(?<![a-zA-Z])\s(?![a-zA-Z])", flags=re.UNICODE)
    re_final_comma = re.compile("\.$")

    with open(filename1 + ".new", 'w') as f1_w:
        with open(filename1, 'r') as f1:
            for index, l1 in enumerate(f1):
                # 重复句的删除
                h = str(hash(l1))
                if h in raw:
                    print(raw[h])
                    print(index)
                    delete += 1
                    continue
                raw[h] = index
                l1 = l1.strip().strip('\n').strip()
                # 全角转半角
                l1 = strQ2B(l1)
                # wmt的处理
                l1 = re_space.sub("", l1)
                l1 = l1.replace(",", u"\uFF0C")
                l1 = re_final_comma.sub(u"\u3002", l1)

                l1_list.append(l1.strip() + '\n')

                if len(l1_list) > 300:
                    f1_w.writelines(l1_list)
                    l1_list = []

        if len(l1_list) > 0:
            f1_w.writelines(l1_list)
    print("delete: ", delete)


def process(filename1, filename2):
    # filename1传汉语，filename2传另外一个语种
    l1_list = []
    l2_list = []
    raw = {}
    delete = 0
    # wmt抄的
    re_space = re.compile(r"(?<![a-zA-Z])\s(?![a-zA-Z])", flags=re.UNICODE)
    re_final_comma = re.compile("\.$")

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
                # 语种检测 不好用！！
                # from langdetect import detect
                # if detect(l1) != 'zh-cn' and detect(l2) != 'en':
                #     print(l1)
                #     print(l2)
                #     continue
                # 语种检测  难用！！
                # import langid
                # if langid.classify(l1)[0] != 'zh' or langid.classify(l2)[0] != 'en':
                #     print(langid.classify(l1)[0]  + l1)
                #     print(langid.classify(l2)[0] + l2)
                l1 = l1.strip().strip('\n').strip()
                l2 = l2.strip().strip('\n').strip()
                # 全角转半角
                l1 = strQ2B(l1)
                l2 = strQ2B(l2)

                # # 大小写转换，后面调用mose的truecase脚本吧
                # l1 = l1.lower()
                # l2 = l2.lower()
                # wmt的处理
                l1 = re_space.sub("", l1)
                l1 = l1.replace(",", u"\uFF0C")
                l1 = re_final_comma.sub(u"\u3002", l1)

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