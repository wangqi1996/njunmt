import mmap
import os

import numpy


# 放回采样
def get_num_of_lines(filename):
    """Get number of lines of a file."""
    f = open(filename, "r")
    buf = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    f.close()
    return lines


def back_sample(src_filename, trg_filename, tag_filename, nums=0, times=1):
    filename_len = get_num_of_lines(src_filename)
    if nums == 0:
        nums = filename_len
    numbers = numpy.random.choice([i for i in range(filename_len)], size=nums, replace=True)
    sorted_numbers = sorted(numbers)
    content1 = []
    content2 = []
    content3 = []
    new_src = os.path.dirname(src_filename) + '/back/uy.back' + str(times)
    new_trg = os.path.dirname(src_filename) + '/back/zh.back' + str(times)
    new_tag = os.path.join(os.path.dirname(src_filename), 'back/tag' + str(times) + '.txt')
    with open(new_src, 'w') as fw_src, open(new_trg, 'w') as fw_trg, open(new_tag, 'w') as fw_tag:
        with open(src_filename, 'r') as fr_src, open(trg_filename, 'r') as fr_trg, open(tag_filename, 'r') as fr_tag:
            index = 0
            for i, (src, trg, tag) in enumerate(zip(fr_src, fr_trg, fr_tag)):
                if index >= nums:
                    break
                while index < nums and sorted_numbers[index] == i:
                    content1.append(src)
                    content2.append(trg)
                    content3.append(tag)
                    index += 1
                if len(content1) > 300:
                    fw_src.writelines(content1)
                    fw_trg.writelines(content2)
                    fw_tag.writelines(content3)
                    content1 = []
                    content2 = []
                    content3 = []
            if len(content1) > 0:
                fw_src.writelines(content1)
                fw_trg.writelines(content2)
    print("done!!!!")


if __name__ == '__main__':
    back_sample("/home/user_data55/wangdq/data/ccmt/uy-zh/sample/uy.token",
                "/home/user_data55/wangdq/data/ccmt/uy-zh/sample/zh.token",
                "/home/user_data55/liuzh/ccmt/bt_attrib_sample600w.txt",
                times=3)
