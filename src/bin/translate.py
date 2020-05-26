import argparse
import os

from src.bin import auto_mkdir
from src.task.nmt import translate

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str,
                    help="""Name of the model.""")

parser.add_argument("--source_path", type=str,
                    help="""Path to source file.""")

parser.add_argument("--model_path", type=str,
                    help="""Path to model files.""")

parser.add_argument("--config_path", type=str,
                    help="""Path to config file.""")

parser.add_argument("--batch_size", type=int, default=5,
                    help="""Batch size of beam search.""")

parser.add_argument("--beam_size", type=int, default=5,
                    help="""Beam size.""")

parser.add_argument("--saveto", type=str,
                    help="""Result prefix.""")

parser.add_argument("--keep_n", type=int, default=-1,
                    help="""To keep how many results. This number should not exceed beam size.""")

parser.add_argument("--use_gpu", action="store_true")

parser.add_argument("--max_steps", type=int, default=150,
                    help="""Max steps of decoding. Default is 150.""")

parser.add_argument("--alpha", type=float, default=-1.0,
                    help="""Factor to do length penalty. Negative value means close length penalty.""")

parser.add_argument("--multi_gpu", action="store_true",
                    help="""Running on multiple GPUs (No need to manually add this option).""")

parser.add_argument("--shared_dir", type=str, default=None,
                    help="""Shared directory across nodes. Default is '/tmp'""")

parser.add_argument("--sample_K", type=int, default=3,
                    help="""sample_K""")

parser.add_argument("--seed", type=int, default=1234,
                    help="""seed""")

parser.add_argument("--copy_head", action="store_true")

parser.add_argument("--ref_path", type=str,
                    help="""ref path""")

parser.add_argument("--num_refs", type=int, default=1,
                    help="""num_refs""")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    auto_mkdir(os.path.dirname(args.saveto))

    translate(args)


if __name__ == '__main__':
    run()
