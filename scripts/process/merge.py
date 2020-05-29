# coding=utf-8

def merge(src, trg, out):
    content = []
    with open(src, 'r') as fs, open(trg, 'r') as ft, open(out, 'w') as fo:
        for lines, linef in zip(fs, ft):
            l = lines.strip() + " ||| " + linef.strip()
            content.append(l + '\n')
            if len(content) > 300:
                fo.writelines(content)
                content = []
        if len(content) > 0:
            fo.writelines(content)


if __name__ == '__main__':
    import sys

    merge(sys.argv[1], sys.argv[2], sys.argv[3])
