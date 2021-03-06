import argparse

from src.bin import auto_mkdir
from src.task.LM import train as LM
from src.task.multi_loss import train as odc_train
from src.task.nmt import train as nmt_train
from src.task.pretrain import train as pretrain
from src.task.select import train as select
from src.task.tune import tune

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str,
                    help="The name of the model. Will alse be the prefix of saving archives.")

parser.add_argument('--reload', action="store_true",
                    help="Whether to restore from the latest archives.")

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--log_path', type=str, default="./log",
                    help="The path for saving tensorboard logs. Default is ./log")

parser.add_argument('--saveto', type=str, default="./save",
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--debug', action="store_true",
                    help="Use debug mode.")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument('--pretrain_path', type=str, default=None, help="The path for pretrained model.")

parser.add_argument("--valid_path", type=str, default="./valid",
                    help="""Path to save translation for bleu evaulation. Default is ./valid.""")

parser.add_argument("--multi_gpu", action="store_true",
                    help="""Running on multiple GPUs (No need to manually add this option).""")

parser.add_argument("--shared_dir", type=str, default="/tmp",
                    help="""Shared directory across nodes. Default is '/tmp'""")

parser.add_argument("--predefined_config", type=str, default=None,
                    help="""Use predefined configuration.""")

parser.add_argument('--display_loss_detail', action="store_true",
                    help="Whether to display loss detail.")

parser.add_argument("--task", type=str, choices=["nmt", "odc", "mlm", "lm", 'tune', 'select'], default="nmt")

# encoder、decoder、generator
parser.add_argument("--pretrain_exclude_prefix", type=str, default=None, help="split by ;")

# encoder_embedding;encoder;decoder_embedding;decoder;encoder_k(encoder_2)
parser.add_argument("--froze_config", type=str, default=None, help="split by ;")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    auto_mkdir(args.log_path)
    auto_mkdir(args.saveto)
    auto_mkdir(args.valid_path)
    if args.task == 'nmt':
        nmt_train(args)
    elif args.task == 'odc':
        odc_train(args)
    elif args.task == 'mlm':
        pretrain(args)
    elif args.task == 'lm':
        LM(args)
    elif args.task == "tune":
        tune(args)
    elif args.task == 'select':
        select(args)
    else:
        raise ValueError("not support task!")


if __name__ == '__main__':
    run()
