# coding=utf-8
def postprocess(f):
    content = []
    with open(f, 'r')  as f1, open(f + '.new', 'w') as f2:
        for line in f1:
            c = line.strip().strip('\n').strip().split(' ')
            content.append(''.join(c) + '\n')
            if len(content) > 300:
                f2.writelines(content)
                content = []
        if len(content) > 0:
            f2.writelines(content)


if __name__ == '__main__':
    import sys

    postprocess(sys.argv[1])
