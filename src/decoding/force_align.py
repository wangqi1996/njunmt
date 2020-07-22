#!/usr/bin/env python

import os
import subprocess
import threading

"""
命令：
 /home/user_data55/wangdq/code/fast_align/build/fast_align -i - -d -T 8.36518 -m 1.14591 -f /home/user_data55/wangdq/data/ccmt/uy-zh/sample/rerank/uy2zh_params
  /home/user_data55/wangdq/code/fast_align/build/fast_align -i - -d -T 11.9333 -m 0.922379 -f /home/user_data55/wangdq/data/ccmt/uy-zh/sample/rerank/zh2uy_params -r
  /home/user_data55/wangdq/code/fast_align/build/atools -i - -j - -c grow-diag-final-and
"""


# Simplified, non-threadsafe version for force_align.py
# Use the version in realtime for development
class Aligner:

    def __init__(self, fwd_params, fwd_err, rev_params, rev_err, heuristic='grow-diag-final-and'):

        build_root = "/home/user_data55/wangdq/code/fast_align/build"
        fast_align = os.path.join(build_root, 'fast_align')
        atools = os.path.join(build_root, 'atools')

        (fwd_T, fwd_m) = self.read_err(fwd_err)
        (rev_T, rev_m) = self.read_err(rev_err)

        self.fwd_cmd = [fast_align, '-i', '-', '-d', '-T', fwd_T, '-m', fwd_m, '-f', fwd_params]
        self.rev_cmd = [fast_align, '-i', '-', '-d', '-T', rev_T, '-m', rev_m, '-f', rev_params, '-r']
        self.tools_cmd = [atools, '-i', '-', '-j', '-', '-c', heuristic]

        # self.fwd_align = popen_io(fwd_cmd)
        # self.rev_align = popen_io(rev_cmd)
        # self.tools = popen_io(tools_cmd)

    def get_align(self, filename):
        score, result = [], []
        with open(filename, 'r') as f:
            for line in f:
                result.append(line.split('|||')[-2].strip())
                score.append(float(line.split('|||')[-1].strip()))
        return result, score

    def new_align(self, token_filename, index):
        fwd_filename = os.path.join(os.path.dirname(token_filename), "align.output" + index)
        rev_filename = os.path.join(os.path.dirname(token_filename), "align.output.rev" + index)
        if not os.path.exists(fwd_filename):
            # 正向
            fwd_cmd = ' '.join(self.fwd_cmd) + ' <' + token_filename + ' ' + '>' + fwd_filename
            print(os.system(fwd_cmd))
        if not os.path.exists(rev_filename):
            # 反向
            rev_cmd = ' '.join(self.rev_cmd) + ' <' + token_filename + ' ' + '>' + rev_filename
            print(os.system(rev_cmd))
        # 对齐
        fwd_align, fwd_score = self.get_align(fwd_filename)
        rev_align, rev_score = self.get_align(rev_filename)
        score_dict = {}
        for seq_num, (fscore, bscore) in enumerate(zip(fwd_score, rev_score)):
            score = (fscore + bscore) / 2
            score_dict.update({
                seq_num: (score,)
            })
        return score_dict

    # def align(self, line):
    #     self.fwd_align.stdin.write('{}\n'.format(line).encode())
    #     # out = self.fwd_align.communicate(line.encode())
    #     self.rev_align.stdin.write('{}\n'.format(line).encode())
    #     # f words ||| e words ||| links ||| score
    #     fwd_line = self.fwd_align.stdout.readline().split('|||')[2].strip()
    #     rev_line = self.rev_align.stdout.readline().split('|||')[2].strip()
    #     self.tools.stdin.write('{}\n'.format(fwd_line).encode())
    #     self.tools.stdin.write('{}\n'.format(rev_line).encode())
    #     al_line = self.tools.stdout.readline().strip()
    #     return al_line

    # def close(self):
    #     self.fwd_align.stdin.close()
    #     self.fwd_align.wait()
    #     self.rev_align.stdin.close()
    #     self.rev_align.wait()
    #     self.tools.stdin.close()
    #     self.tools.wait()

    def read_err(self, err):
        (T, m) = ('', '')
        for line in open(err):
            # expected target length = source length * N
            if 'expected target length' in line:
                m = line.split()[-1]  # 1.14591
            # final tension: N
            elif 'final tension' in line:
                T = line.split()[-1]  # 8.36518
        return (T, m)


def popen_io(cmd):
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def consume(s):
        for _ in s:
            pass

    threading.Thread(target=consume, args=(p.stderr,)).start()
    return p


def force_align_main(fwd_params, fwd_err, rev_params, rev_err, token_filename, index):
    aligner = Aligner(fwd_params, fwd_err, rev_params, rev_err, )

    score_dict = aligner.new_align(token_filename, index)
    return score_dict
