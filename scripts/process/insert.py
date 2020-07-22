def insert(filename):
    # 维汉
    id_list = [323, 335, 1554, 1836, 1957, 2019, 2183, 2935, 3387, 3919, 4265, 5490, 2785]
    c = "\n"
    content = []
    index = 0
    with open(filename, 'r') as f:
        for l in f:
            if (index + 1) in id_list:
                content.append(c)
                index += 1
            content.append(l)
            index += 1
    with open(filename, 'w') as f:
        f.writelines(content)


def delete(filename):
    id_list = [2778]
    content = []
    with open(filename, 'r') as f:
        for index, l in enumerate(f):
            if (index + 1) in id_list:
                continue
            content.append(l)
    with open(filename, 'w') as f:
        f.writelines(content)


if __name__ == '__main__':
    # import sys
    #
    # filename = sys.argv[1]
    # insert(filename)
    # trg_filename_list = [
    #     "/home/user_data55/wangdq/data/ccmt/uy-zh/output/rerank/ensemble_all_sample_relative_ave.out." + str(
    #         i) for
    #     i in range(24)]
    #
    # for filename in trg_filename_list:
    #     delete(filename)
    # delete("/home/user_data55/wangdq/data/ccmt/uy-zh/test/test.txt")