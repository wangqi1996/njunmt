# coding=utf-8

def reverse(filename):
    new_filename = filename + '.reverse'
    content = []
    with open(new_filename, 'w') as f2:
        with open(filename, 'r') as f:
            for line in f:
                c = line.strip().split(' ')
                c.reverse()
                content.append(' '.join(c) + '\n')

                if len(content) > 300:
                    f2.writelines(content)
                    content = []
            if len(content) > 0:
                f2.writelines(content)


def join(filename):
    new_filename = filename + '.join'
    content = []
    with open(new_filename, 'w') as f2:
        with open(filename, 'r') as f:
            for line in f:
                c = line.strip().split(' ')
                content.append(''.join(c) + '\n')

                if len(content) > 300:
                    f2.writelines(content)
                    content = []
            if len(content) > 0:
                f2.writelines(content)

if __name__ == '__main__':
    import sys
    #
    # reverse(sys.argv[1])
    reverse("/home/user_data55/wangdq/data/ccmt/uy-zh/sample3/zh.token")
    # join("/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/reverse/dev.zh.token.reverse")
