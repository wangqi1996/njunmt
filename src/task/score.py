import itertools
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

import src.distributed as dist
from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.vocabulary import Vocabulary
from src.models import build_model, load_predefined_configs
from src.modules.criterions import NMTCriterion
from src.utils.common_utils import *
from src.utils.configs import pretty_configs, add_user_configs, default_base_configs
from src.utils.logging import *


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def load_model_parameters(path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)

    if "model" in state_dict:
        return state_dict["model"]
    return state_dict


def combine_from_all_shards(all_gathered_output):
    """Combine all_gathered output split by ```split_shards_iterator```
    """
    output = []
    for items in itertools.zip_longest(*all_gathered_output, fillvalue=None):
        for item in items:
            if item is not None:
                output.append(item)

    return output


def prepare_configs(config_path: str, predefined_config: str = "") -> dict:
    """Prepare configuration file"""
    # 1. Default configurations
    default_configs = default_base_configs()

    # 2. [Optional] Load pre-defined configurations
    default_configs = load_predefined_configs(default_configs, predefined_config)

    # 3. Load user configs
    config_path = os.path.abspath(config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    configs = add_user_configs(default_configs, configs)

    return configs


def loss_evaluation(model, critic, valid_iterator, rank=0, world_size=1):
    """
    :type model: Transformer

    :type critic: NMTCriterion

    :type valid_iterator: DataIterator
    """

    n_sents = 0

    sum_loss = 0.0

    valid_iter = valid_iterator.build_generator()

    for batch in valid_iter:
        _, seqs_x, seqs_y = batch

        n_sents += len(seqs_x)

        x, y = prepare_data(seqs_x, seqs_y, cuda=Constants.USE_GPU)

        loss = compute_forward(model=model,
                               critic=critic,
                               seqs_x=x,
                               seqs_y=y,
                               eval=True)

        if np.isnan(loss):
            WARN("NaN detected!")

        sum_loss += float(loss)

    if world_size > 1:
        sum_loss = dist.all_reduce_py(sum_loss)
        n_sents = dist.all_reduce_py(n_sents)

    return float(sum_loss / n_sents)


def prepare_data(seqs_x, seqs_y, cuda=False, batch_first=True):
    """
    Args:
        eval ('bool'): indicator for eval/infer.

    Returns:

    """

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    seqs_y = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    return x, y


def compute_forward(model,
                    critic,
                    seqs_x,
                    seqs_y,
                    eval=False,
                    normalization=1.0,
                    norm_by_words=False
                    ):
    """
    :type model: nn.Module

    :type critic: NMTCriterion
    """
    y_inp = seqs_y[:, :-1].contiguous()
    y_label = seqs_y[:, 1:].contiguous()

    model.eval()
    # For compute loss
    with torch.no_grad():
        log_probs = model(seqs_x, y_inp)

        batch_size, seq_len, vocab_size = log_probs.shape
        log_probs = log_probs.view(-1, vocab_size)
        y_label = y_label.view(-1)

        loss = F.nll_loss(log_probs, y_label, reduce=False, ignore_index=Constants.PAD)
        loss = loss.view(batch_size, seq_len)
        loss = loss.sum(-1)

        y_label = y_label.view(batch_size, seq_len)
        valid_token = (y_label != Constants.PAD).sum(-1)
        loss = loss.double().div(valid_token.double())

    return loss


def load_pretrained_model(nmt_model, pretrain_path, device, exclude_prefix=None):
    """
    Args:
        nmt_model: model.
        pretrain_path ('str'): path to pretrained model.
        map_dict ('dict'): mapping specific parameter names to those names
            in current model.
        exclude_prefix ('dict'): excluding parameters with specific names
            for pretraining.

    Raises:
        ValueError: Size not match, parameter name not match or others.

    """
    if exclude_prefix is None:
        exclude_prefix = []
    else:
        exclude_prefix = exclude_prefix.split(";")
    if pretrain_path is not None:
        INFO("Loading pretrained model from {}".format(pretrain_path))

        all_parameter_names = set([name for name in nmt_model.state_dict().keys()])

        pretrain_params = torch.load(pretrain_path, map_location=device)
        for name, params in pretrain_params.items():

            if name not in all_parameter_names:
                continue

            flag = False
            for pp in exclude_prefix:
                if name.startswith(pp):
                    flag = True
                    break
            if flag:
                continue

            INFO("Loading param: {}...".format(name))
            try:
                nmt_model.load_state_dict({name: params}, strict=False)
            except Exception as e:
                WARN("{}: {}".format(str(Exception), e))

        INFO("Pretrained model loaded.")


def train(config_path, model_path, model_type, src_filename, trg_filename):
    """
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = True
    print(config_path)
    print(model_path)
    print(model_type)

    world_size = 1
    rank = 0
    local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(config_path)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    training_configs = configs['training_configs']

    INFO(pretty_configs(configs))

    Constants.SEED = training_configs['seed']
    set_seed(Constants.SEED)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=src_filename,
                        vocabulary=vocab_src,
                        max_len=100,
                        is_train_dataset=False,
                        ),
        TextLineDataset(data_path=trg_filename,
                        vocabulary=vocab_tgt,
                        is_train_dataset=False,
                        max_len=100,
                        )
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=20,
                                  use_bucket=training_configs['use_bucket'],
                                  buffer_size=training_configs['buffer_size'],
                                  numbering=True,
                                  world_size=world_size,
                                  rank=rank)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, vocab_src=vocab_src,
                            **model_configs)
    INFO(nmt_model)

    # 2. Move to GPU
    if Constants.USE_GPU:
        nmt_model = nmt_model.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(nmt_model, model_path, device=Constants.CURRENT_DEVICE)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Prepare training

    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    grad_denom = 0
    train_loss = 0.0
    cum_n_words = 0
    valid_loss = best_valid_loss = float('inf')

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    INFO('Begin training...')
    eidx = 0
    uidx = 0
    score_result = dict()

    # Build iterator and progress bar
    training_iter = valid_iterator.build_generator()

    training_progress_bar = tqdm(desc=' - (Epc {}, Upd {}) '.format(eidx, uidx),
                                 total=len(valid_iterator),
                                 unit="sents"
                                 )

    for batch in training_iter:
        seqs_numbers, seqs_x, seqs_y = batch

        batch_size = len(seqs_x)
        cum_n_words += sum(len(s) for s in seqs_y)

        try:
            # Prepare data
            x, y = prepare_data(seqs_x, seqs_y, cuda=Constants.USE_GPU)

            y_inp = y[:, :-1].contiguous()
            y_label = y[:, 1:].contiguous()  # [batch_size, seq_len]
            log_probs = nmt_model(x, y_inp, log_probs=True)  # [batch_size, seq_len, vocab_size]

            _, seq_len = y_label.shape
            log_probs = log_probs.view(-1, vocab_tgt.max_n_words)
            y_label = y_label.view(-1)
            loss = F.nll_loss(log_probs, y_label, reduce=False, ignore_index=vocab_tgt.pad)
            loss = loss.view(batch_size, seq_len)
            loss = loss.sum(-1)

            y_label = y_label.view(batch_size, seq_len)
            valid_token = (y_label != vocab_tgt.pad).sum(-1)
            loss = loss.double().div(valid_token.double())
            for seq_num, l in zip(seqs_numbers, loss):
                assert seq_num not in score_result
                score_result.update({seq_num: l.item()})

            uidx += 1
            grad_denom += batch_size

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
            else:
                raise e

        if training_progress_bar is not None:
            training_progress_bar.update(batch_size)
            training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

            postfix_str = 'TrainLoss: {:.2f}, ValidLoss(best): {:.2f} ({:.2f}), '.format(train_loss, valid_loss,
                                                                                         best_valid_loss)
            training_progress_bar.set_postfix_str(postfix_str)

    training_progress_bar.close()
    return score_result


if __name__ == '__main__':
    # rerank_max(beam_size=5, outputs="/home/user_data55/wangdq/code/njunmt/log/zh2en/outputmt03.a")
    rerank_config = [
        {"config_path": "/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/transformer_uy2zh_bpe_new4.yaml",
         "model_path": "/home/user_data55/wangdq/code/njunmt_dist/save/uy2zh/uy2zh_base_d4/transformer.best.final",
         "model_type": "l2r"},
        {"config_path": "/home/user_data55/wangdq/code/njunmt_dist/configs/zh2uy/transformer_zh2uy_bpe_new2.yaml",
         "model_path": "/home/wangdq/save/zh2uy/zh2uy_base_a/transformer.best.final",
         "model_type": "t2s"},
    ]
    src_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/monolingual/all/uy.token"
    trg_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/monolingual/all/zh.token"
    # src_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/parallel/uy.token"
    # trg_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/parallel/zh.token"
    import sys

    import numpy as np
    if sys.argv[1] == "l2r":
        print("l2r!!!!")
        score_dict = train(src_filename=src_filename, trg_filename=trg_filename, **rerank_config[0])
        np.save(src_filename + ".l2r_score", score_dict)
    elif sys.argv[1] == "t2s":
        print("t2s!!!!!!")
        score_dict = train(src_filename=src_filename, trg_filename=trg_filename, **rerank_config[1])
        np.save(src_filename + ".t2s_score", score_dict)


