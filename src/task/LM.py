import itertools
import os
import random
import time

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset
from src.data.vocabulary import Vocabulary
from src.models import build_model, load_predefined_configs
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

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
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

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos

    train_bitext_dataset = TextLineDataset(data_path=data_configs['train_data'][0],
                                           vocabulary=vocab_src,
                                           max_len=data_configs['max_len'][0],
                                           is_train_dataset=True
                                           )

    valid_bitext_dataset = TextLineDataset(data_path=data_configs['valid_data'][0],
                                           vocabulary=vocab_src,
                                           is_train_dataset=False
                                           )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank)
    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=True, buffer_size=100000, numbering=True,shuffle=False,
                                  world_size=world_size, rank=rank)

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

    model_collections = Collections()

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(flags.saveto, flags.model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    best_model_prefix = os.path.join(flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    best_model_saver = Saver(save_prefix=best_model_prefix, num_max_keeping=training_configs['num_kept_best_model'])

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(vocab_size=vocab_src.max_n_words, padding_idx=vocab_src.pad, vocab_src=vocab_src,
                            **model_configs)
    INFO(nmt_model)
    # 损失函数
    critic = torch.nn.CrossEntropyLoss(ignore_index=Constants.PAD)
    INFO(critic)

    # 2. Move to GPU
    if Constants.USE_GPU:
        nmt_model = nmt_model.cuda()
        critic = critic.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(nmt_model, flags.pretrain_path, exclude_prefix=flags.pretrain_exclude_prefix,
                          device=Constants.CURRENT_DEVICE)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optimizer = torch.optim.Adam(nmt_model.parameters(), lr=optimizer_configs['learning_rate'])

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    is_early_stop = model_collections.get_collection("is_early_stop", [False, ])[-1]

    train_loss_meter = AverageMeter()
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    grad_denom = 0
    train_loss = 0.0
    cum_n_words = 0
    valid_loss = best_valid_loss = float('inf')

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    INFO('Begin training...')

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc=' - (Epc {}, Upd {}) '.format(eidx, uidx),
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None

        for batch in training_iter:
            seqs_x = batch

            batch_size = len(seqs_x)
            cum_n_words = 0.0
            train_loss = 0.0

            try:
                # Prepare data
                grad_denom += batch_size
                x = prepare_data(seqs_x, seqs_y=None, cuda=Constants.USE_GPU)
                nmt_model.train()
                critic.train()
                critic.zero_grad()
                with torch.enable_grad():
                    logits = nmt_model(x[:-1])
                    logits = logits.view(-1, vocab_src.max_n_words)
                    trg = x[1:]
                    trg = trg.view(-1)
                    loss = critic(logits, trg)
                    loss.backward()
                    optimizer.step()
                    valid_token = (trg != Constants.PAD).long().sum().item()
                    cum_n_words += valid_token
                    train_loss += loss.item() * valid_token

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom, update uidx
            # - learning rate scheduling
            # - update moving average

            if training_progress_bar is not None:
                training_progress_bar.update(grad_denom)
                training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                postfix_str = 'TrainLoss: {:.2f}, ValidLoss(best): {:.2f} ({:.2f}), '.format(train_loss / cum_n_words,
                                                                                             valid_loss,
                                                                                             best_valid_loss)
                training_progress_bar.set_postfix_str(postfix_str)

            # 4. update meters
            train_loss_meter.update(train_loss, cum_n_words)
            sent_per_sec_meter.update(grad_denom)
            tok_per_sec_meter.update(cum_n_words)

            # 5. reset accumulated variables, update uidx
            grad_denom = 0
            uidx += 1
            cum_n_words = 0.0
            train_loss = 0.0

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss", scalar_value=train_loss_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       min_step=training_configs['bleu_valid_warmup'],
                                       debug=flags.debug):

                valid_iter = valid_iterator.build_generator()
                valid_loss = 0
                total_tokens = 0
                for batch in valid_iter:
                    seq_number, seqs_x = batch
                    x = prepare_data(seqs_x, seqs_y=None, cuda=Constants.USE_GPU)
                    nmt_model.eval()
                    critic.eval()
                    with torch.no_grad():
                        logits = nmt_model(x[:-1])
                        logits = logits.view(-1, vocab_src.max_n_words)
                        trg = x[1:]
                        valid_token = (trg != Constants.PAD).sum(-1)
                        batch_size, seq_len = trg.shape
                        trg = trg.view(-1)
                        # loss = critic(logits, trg)
                        # valid_token = (trg != Constants.PAD).long().sum().item()
                        # total_tokens += valid_token
                        # valid_loss += loss.item() * valid_token
                        import torch.nn.functional as F
                        loss = F.cross_entropy(logits, trg, reduce=False, ignore_index=vocab_src.pad)
                        loss = loss.view(batch_size, seq_len)
                        loss = loss.sum(-1)
                        print(seq_number)
                        print(loss.double().div(valid_token.double()))
                exit(0)
                valid_loss = valid_loss / total_tokens
                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()
                best_valid_loss = min_history_loss
                if summary_writer is not None:
                    summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                    summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)
                # If model get new best valid bleu score
                if valid_loss <= best_valid_loss:
                    bad_count = 0

                    if is_early_stop is False:
                        if rank == 0:
                            # 1. save the best model
                            torch.save(nmt_model.state_dict(), best_model_prefix + ".final")

                            # 2. record all several best models
                            best_model_saver.save(global_step=uidx,
                                                  model=nmt_model,
                                                  optimizer=optimizer,
                                                  collections=model_collections)
                    else:
                        bad_count += 1

                        # At least one epoch should be traversed
                        if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                            is_early_stop = True
                            WARN("Early Stop!")
                            exit(0)

                if summary_writer is not None:
                    summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f}  patience: {2}".format(
                    uidx, valid_loss, bad_count
                ))

            # ================================================================================== #
            # # Saving checkpoints
            # if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
            #     model_collections.add_to_collection("uidx", uidx)
            #     model_collections.add_to_collection("eidx", eidx)
            #     model_collections.add_to_collection("bad_count", bad_count)
            #
            #     if not is_early_stop:
            #         if rank == 0:
            #             checkpoint_saver.save(global_step=uidx,
            #                                   model=nmt_model,
            #                                   optim=optimizer,
            #                                   collections=model_collections)

        if training_progress_bar is not None:
            training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break
