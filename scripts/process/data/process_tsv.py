# coding=utf-8
# import langid

def process_tsv(filename):
    zh_content = []
    en_content = []
    with open(filename, 'r') as f:
        with open(filename + ".zh", 'w') as f1, open(filename + ".en", 'w') as f2:
            for index, line in enumerate(f):
                line = line.strip()
                if len(line) <= 0:
                    continue
                c = line.split('\t')
                if len(c) != 2:
                    print(index , line)
                    continue
                en, zh = c
                en = en.strip()
                zh = zh.strip()
                if len(en) <= 0 or len(zh) <= 0:
                    print(line)
                    continue
                # # 语种检测
                # _en = langid.classify(en)
                # _zh = langid.classify(zh)
                # if _en != 'en' or _zh != 'zh':
                #     print(en)
                #     print(zh)
                #     continue

                en_content.append(en + '\n')
                zh_content.append(zh + '\n')
                if len(en_content) >300:
                    f2.writelines(en_content)
                    f1.writelines(zh_content)
                    en_content = []
                    zh_content = []

            if len(en_content) > 0:
                f2.writelines(en_content)
                f1.writelines(zh_content)

if __name__ == '__main__':
    filename = "/home/user_data55/wangdq/data/wmt2020/paralle/data/wikititiles.tsv"
    process_tsv(filename)