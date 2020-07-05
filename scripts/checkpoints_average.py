# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import collections
import os
import torch
from typing import List


def load_model_parameters(path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)

    if "model" in state_dict:
        return state_dict["model"]
    return state_dict

# 直接平均参数
def average_checkpoints(checkpoints_path: List):
    ave_state_dict = collections.OrderedDict()
    param_names = None

    for ii, f in enumerate(checkpoints_path):
        state_dict = load_model_parameters(f)

        if param_names is None:
            param_names = list(state_dict.keys())

        if param_names != list(state_dict.keys()):
            raise KeyError(
                "Checkpoint {0} has inconsistent parameters".format(f)
            )

        for k in param_names:
            if k not in ave_state_dict:
                ave_state_dict[k] = state_dict[k]
            else:
                ave_state_dict[k] = (ave_state_dict[k] * ii + state_dict[k]) / float(ii + 1)

    return ave_state_dict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_dir", type=str,
                        default="/home/user_data55/wangdq/save/uy2zh/final/uy2zh_sample_big_ave",
                        help="Directory to store checkpoints")

    parser.add_argument("--checkpoints", type=str, nargs="+",
                        default="transformer.best.101500,transformer.best.115500,transformer.best.119000,transformer.best.129500,transformer.best.140000,transformer.best.150500",
                        # default="transformer.ckpt.199500,transformer.ckpt.203000,transformer.ckpt.206500,transformer.ckpt.210000,transformer.ckpt.213500,transformer.ckpt.217000",
                        help="Names of checkpoint files to be averaged.")

    parser.add_argument("--saveto", type=str,
                        default="/home/user_data55/wangdq/save/uy2zh/final/uy2zh_sample_big_ave/checkpoint.best.ave",
                        help="Saving path of averaged checkpoint (only model parameters).")

    args = parser.parse_args()

    return args


def main(args):
    checkpoints_path = [os.path.join(args.checkpoint_dir, ckpt) for ckpt in args.checkpoints.split(',')]

    ave_state_dict = average_checkpoints(checkpoints_path)

    torch.save(ave_state_dict, args.saveto)


if __name__ == '__main__':
    args = parse_args()
    main(args)



# "transformer.best.101500,transformer.best.115500,transformer.best.119000,transformer.best.129500,transformer.best.140000,transformer.best.150500