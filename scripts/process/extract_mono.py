# coding=utf-8

import os


def extract_mono(filename):
    processed_name = os.path.join(os.path.dirname(filename), 'xmu.txt2')
    import re
    start_pattern = re.compile(r'<text')
    end_pattern = re.compile(r'</text>')
    content_pattern = re.compile(r'<text .*>(.*?)</text>')
    contents = []
    raw_text = ''
    start = False
    with open(filename, 'r') as f:
        with open(processed_name, 'w') as fw:
            for line in f:
                end_search = re.search(end_pattern, line)
                start_search = re.search(start_pattern, line)
                if start is True:
                    assert start_search is None, ""
                    raw_text += (line.strip('\n').strip())
                    if end_search is not None:
                        text = re.findall(content_pattern, raw_text)
                        contents.append(text[0] + '\n')
                        start = False
                        if len(contents) > 300:
                            fw.writelines(contents)
                            contents = []
                else:
                    assert end_search is None, u''
                    if start_search is not None:
                        raw_text = line.strip('\n').strip(' ')
                        start = True

            if len(contents) >= 0:
                fw.writelines(contents)

if __name__ == '__main__':
    import sys
    extract_mono(sys.argv[1])