import os
import random

import numpy as np
import torch
import yaml

from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset, AttributeDataset
from src.data.vocabulary import Vocabulary
from src.decoding.utils import tensor_gather_helper2
from src.models import load_predefined_configs, build_model
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


def prepare_data(seqs_x, seqs_y, cuda=False, batch_first=True, bt_attrib=None):
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

    # add bt tag
    if bt_attrib is not None and Constants.USE_BT and Constants.USE_BTTAG:
        seqs_x = list(map(lambda s, is_bt: [Constants.BTTAG] + s if is_bt[0] else s, seqs_x, bt_attrib))

    x = _np_pad_batch_2D(samples=seqs_x, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    seqs_y = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    return x, y


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


def train(flags):
    Constants.USE_GPU = flags.use_gpu
    local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    test_mean_hidden, test_seq_numbers = test_data(flags)

    encoder_filename = "/home/wangdq/encoder.mean.output"
    seq_numbers_filename = '/home/wangdq/seq_numbers.output'

    all_seq_numbers_list = []
    with open(seq_numbers_filename, 'r') as f_seq_numbers:
        for line in f_seq_numbers:  # has only one line
            all_seq_numbers_list = [int(i) for i in line.strip().split(' ')]

    select_sentences = set()

    previous_min_distance = None
    previous_seq_numbers = None
    top_numbers = 200
    all_numbers = 6340403
    times = 30

    slice = int(all_numbers / times) + 10
    f_encoder = open(encoder_filename, 'r')
    i = 0
    start_index = i * slice
    end_index = (i + 1) * slice
    all_mean_encoder_hidden = []
    print(start_index)
    print(end_index)
    end_index = min(end_index, all_numbers - 1)

    seq_index = None
    for index, line in enumerate(f_encoder):

        if index < start_index:
            continue
        elif index < end_index:
            c = [float(i) for i in line.strip().split(' ')]
            all_mean_encoder_hidden.append(c)
        else:
            c = [float(i) for i in line.strip().split(' ')]
            all_mean_encoder_hidden.append(c)
            all_seq_numbers = all_seq_numbers_list[start_index: end_index + 1]
            all_mean_encoder_hidden = torch.FloatTensor(all_mean_encoder_hidden).cuda()
            all_seq_numbers = torch.Tensor(all_seq_numbers).int().cpu()
            INFO("train end!!!")
            # 计算test集合每个句子的表示： mean pool

            distance = torch.matmul(test_mean_hidden,
                                    all_mean_encoder_hidden.transpose(0, 1))  # [batch, train_batch]
            all_mean_encoder_hidden = []
            b, _ = distance.shape

            if i != 0:
                distance = torch.cat((previous_min_distance, distance), dim=-1)
                all_seq_numbers = torch.cat(
                    (previous_seq_numbers, all_seq_numbers.unsqueeze(0).repeat(b, 1)), dim=-1)
            else:
                all_seq_numbers = all_seq_numbers.unsqueeze(0).repeat(b, 1)

            torch.cuda.empty_cache()

            _distance, indices = distance.topk(top_numbers, dim=-1, largest=True, sorted=False)

            seq_index = tensor_gather_helper2(gather_indices=indices.cpu(),
                                              gather_from=all_seq_numbers,
                                              batch_size=b,
                                              beam_size=all_seq_numbers.shape[1],
                                              gather_shape=[-1]).view(b, top_numbers)

            previous_min_distance = _distance
            previous_seq_numbers = seq_index

            i += 1
            start_index = end_index + 1
            end_index = (i + 1) * slice
            print(start_index)
            print(end_index)
            end_index = min(end_index, all_numbers - 1)
            all_mean_encoder_hidden = []

    select_sentences.update(seq_index.view(-1).int().cpu().tolist())
    f_encoder.close()
    select_filename = 'select.filename'
    with open(select_filename, 'w') as f:
        content = [str(i) for i in select_sentences]
        f.writelines(' '.join(content) + '\n')

    print("write into file")
    # 根据select_sentences 构造训练集合
    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    src_filename = data_configs['train_data'][0]
    trg_filename = data_configs['train_data'][1]
    new_src_filename = src_filename + '.select'
    new_trg_filename = trg_filename + '.select'
    src_content, trg_content = [], []
    with open(new_src_filename, 'w') as new_src, open(new_trg_filename, 'w') as new_trg:
        with open(src_filename, 'r') as src, open(trg_filename, 'r') as trg:
            for index, (s, t) in enumerate(zip(src, trg)):
                if index in select_sentences:
                    src_content.append(s)
                    trg_content.append(t)
                    if len(src_content) > 300:
                        new_src.writelines(src_content)
                        new_trg.writelines(trg_content)
                        src_content = []
                        trg_content = []
        if len(src_content) > 0:
            new_src.writelines(src_content)
            new_trg.writelines(trg_content)


def test_data(flags):
    Constants.USE_GPU = flags.use_gpu

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

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    training_configs = configs['training_configs']
    bt_configs = configs['bt_configs'] if 'bt_configs' in configs else None
    if bt_configs is not None:
        print("btconfigs ", bt_configs)
        if 'bt_attribute_data' not in bt_configs:
            Constants.USE_BT = False
            bt_configs = None
        else:
            Constants.USE_BT = True
            Constants.USE_BTTAG = bt_configs['use_bttag']
            Constants.USE_CONFIDENCE = bt_configs['use_confidence']
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
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        is_train_dataset=False,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        is_train_dataset=False
                        )
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=True, buffer_size=100000, numbering=True,
                                  world_size=world_size, rank=rank, shuffle=False)

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
                            vocab_tgt=vocab_tgt,
                            **model_configs)
    INFO(nmt_model)

    # 2. Move to GPU
    if Constants.USE_GPU:
        nmt_model = nmt_model.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(nmt_model, flags.pretrain_path, exclude_prefix=flags.pretrain_exclude_prefix,
                          device=Constants.CURRENT_DEVICE)
    nmt_model = nmt_model.encoder
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin training...')
    # 计算train集合每个句子的表示：mean pool
    nmt_model.eval()

    # 计算test集合每个句子的表示： mean pool
    valid_iter = valid_iterator.build_generator()
    all_seq_numbers = []

    all_mean_encoder_hidden = None
    for batch in valid_iter:
        bt_attrib = None
        seq_numbers, seqs_x, seqs_y = batch
        all_seq_numbers.extend(seq_numbers)
        x = prepare_data(seqs_x, seqs_y=None, cuda=Constants.USE_GPU, bt_attrib=bt_attrib)
        try:
            with torch.no_grad():
                encoder_hidden, mask = nmt_model(x)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
            else:
                raise e

        valid_hidden = (mask == False).float().cuda()
        sum_encoder_hidden = (encoder_hidden * valid_hidden.unsqueeze(-1)).sum(dim=1)
        valid_tokens = (mask == False).sum(-1)
        mean_encoder_hidden = sum_encoder_hidden.float().div(valid_tokens.unsqueeze(1))

        if all_mean_encoder_hidden is None:
            all_mean_encoder_hidden = mean_encoder_hidden
        else:
            all_mean_encoder_hidden = torch.cat((all_mean_encoder_hidden, mean_encoder_hidden), dim=0)
    return all_mean_encoder_hidden, all_seq_numbers


def train2(flags):
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
    Constants.USE_GPU = flags.use_gpu

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

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    training_configs = configs['training_configs']
    bt_configs = configs['bt_configs'] if 'bt_configs' in configs else None
    if bt_configs is not None:
        print("btconfigs ", bt_configs)
        if 'bt_attribute_data' not in bt_configs:
            Constants.USE_BT = False
            bt_configs = None
        else:
            Constants.USE_BT = True
            Constants.USE_BTTAG = bt_configs['use_bttag']
            Constants.USE_CONFIDENCE = bt_configs['use_confidence']
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

    # bt tag dataset
    if Constants.USE_BT:
        if Constants.USE_BTTAG:
            Constants.BTTAG = vocab_src.bttag
        train_bitext_dataset = ZipDataset(
            TextLineDataset(data_path=data_configs['train_data'][0],
                            vocabulary=vocab_src,
                            max_len=data_configs['max_len'][0],
                            is_train_dataset=True
                            ),
            TextLineDataset(data_path=data_configs['train_data'][1],
                            vocabulary=vocab_tgt,
                            max_len=data_configs['max_len'][1],
                            is_train_dataset=True
                            ),
            AttributeDataset(data_path=bt_configs['bt_attribute_data'], is_train_dataset=True)
        )
    else:
        train_bitext_dataset = ZipDataset(
            TextLineDataset(data_path=data_configs['train_data'][0],
                            vocabulary=vocab_src,
                            max_len=data_configs['max_len'][0],
                            is_train_dataset=True
                            ),
            TextLineDataset(data_path=data_configs['train_data'][1],
                            vocabulary=vocab_tgt,
                            max_len=data_configs['max_len'][1],
                            is_train_dataset=True
                            )
        )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size, numbering=True,
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
                            vocab_tgt=vocab_tgt,
                            **model_configs)
    INFO(nmt_model)

    # 2. Move to GPU
    if Constants.USE_GPU:
        nmt_model = nmt_model.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(nmt_model, flags.pretrain_path, exclude_prefix=flags.pretrain_exclude_prefix,
                          device=Constants.CURRENT_DEVICE)
    nmt_model = nmt_model.encoder
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin training...')

    # 计算train集合每个句子的表示：mean pool
    training_iter = training_iterator.build_generator()
    nmt_model.eval()

    all_seq_numbers = []
    encoder_filename = "/home/wangdq/encoder.mean.output"
    seq_numbers_filename = '/home/wangdq/seq_numbers.output'

    processd = 0

    with open(encoder_filename, 'w') as f_encoder, open(seq_numbers_filename, 'w') as f_seq_numbers:

        for batch in training_iter:
            bt_attrib = None
            # bt attrib data
            if Constants.USE_BT:
                seq_numbers, seqs_x, seqs_y, bt_attrib = batch  # seq_numerbs从0开始编号
            else:
                seq_numbers, seqs_x, seqs_y = batch

            x = prepare_data(seqs_x, seqs_y=None, cuda=Constants.USE_GPU, bt_attrib=bt_attrib)

            try:
                with torch.no_grad():
                    encoder_hidden, mask = nmt_model(x)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                else:
                    raise e

            valid_hidden = (mask == False).float().cuda()
            sum_encoder_hidden = (encoder_hidden * valid_hidden.unsqueeze(-1)).sum(dim=1)
            valid_tokens = (mask == False).sum(-1)
            mean_encoder_hidden = sum_encoder_hidden.float().div(valid_tokens.unsqueeze(1))

            all_seq_numbers.extend(seq_numbers)
            # if all_mean_encoder_hidden is None:
            #     all_mean_encoder_hidden = mean_encoder_hidden.cpu()
            # else:
            #     all_mean_encoder_hidden = torch.cat((all_mean_encoder_hidden, mean_encoder_hidden.cpu()), dim=0)

            mean_encoder_list = mean_encoder_hidden.cpu().numpy().tolist()
            content = [[str(i) for i in mean] for mean in mean_encoder_list]
            content = [' '.join(mean) + '\n' for mean in content]
            f_encoder.writelines(content)

            processd += len(seq_numbers)
            print(processd)

        content = [str(i) for i in all_seq_numbers]
        content = ' '.join(content)
        f_seq_numbers.writelines(content)
