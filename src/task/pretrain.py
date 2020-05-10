# coding=utf-8

import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import src.distributed as dist
from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.vocabulary import Vocabulary
from src.models import build_model
from src.modules.criterions import NMTCriterion
from src.optim import Optimizer
from src.optim.lr_scheduler import build_scheduler
from src.task.nmt import prepare_configs, load_pretrained_model, prepare_data, set_seed
from src.utils.common_utils import *
from src.utils.configs import pretty_configs
from src.utils.logging import *
from src.utils.moving_average import MovingAverage


def compute_accuray(log_probs, sentences):
    # log_probs: [batch_size, seq_len, vocab_size]
    # sentences: [batch_size, seq_len]
    mask = (sentences != 0) * (sentences != 2) * (sentences != 1)
    predicts = log_probs.argmax(dim=-1)
    predicts = predicts.masked_select(mask)
    sentences = sentences.masked_select(mask)
    total_tokens = predicts.size(0)
    correct_tokens = (predicts == sentences).long().sum().item()
    return total_tokens, correct_tokens


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

    words_norm = seqs_y.ne(Constants.PAD).float().sum(1)

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs = model(seqs_x, seqs_y)
            total_tokens, correct_tokens = compute_accuray(log_probs, seqs_y)
            loss = critic(inputs=log_probs, labels=seqs_y, reduce=False, normalization=normalization)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()
        torch.autograd.backward(loss)
        return loss.item(), total_tokens, correct_tokens
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs = model(seqs_x, seqs_y)
            total_tokens, correct_tokens = compute_accuray(log_probs, seqs_y)
            loss = critic(inputs=log_probs, labels=seqs_y, normalization=normalization, reduce=True)
        return loss.item(), total_tokens, correct_tokens


def loss_evaluation(model, critic, valid_iterator, rank=0, world_size=1):
    """
    :type model: Transformer

    :type critic: NMTCriterion

    :type valid_iterator: DataIterator
    """

    n_sents = 0

    sum_loss = 0.0

    valid_iter = valid_iterator.build_generator()
    total_tokens = 0
    correct_tokens = 0
    for batch in valid_iter:
        _, seqs_x, seqs_y = batch

        n_sents += len(seqs_x)

        x, y = prepare_data(seqs_x, seqs_y, cuda=Constants.USE_GPU)

        loss, t, c = compute_forward(model=model,
                                     critic=critic,
                                     seqs_x=x,
                                     seqs_y=y,
                                     eval=True)

        if np.isnan(loss):
            WARN("NaN detected!")

        sum_loss += float(loss)
        total_tokens += t
        correct_tokens += c

    if world_size > 1:
        sum_loss = dist.all_reduce_py(sum_loss)
        n_sents = dist.all_reduce_py(n_sents)
        total_tokens = dist.all_reduce_py(total_tokens)
        correct_tokens = dist.all_reduce_py(correct_tokens)

    return float(sum_loss / n_sents), total_tokens / correct_tokens


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

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
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
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos

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

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=True, buffer_size=100000, numbering=True,
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

    lrate = optimizer_configs['learning_rate']
    model_collections = Collections()

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(flags.saveto, flags.model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    best_model_prefix = os.path.join(flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    best_model_saver = Saver(save_prefix=best_model_prefix, num_max_keeping=training_configs['num_kept_best_model'])

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, vocab_src=vocab_src,
                            **model_configs)
    INFO(nmt_model)

    critic = NMTCriterion(label_smoothing=model_configs['label_smoothing'], padding_idx=vocab_tgt.pad)

    INFO(critic)

    # 2. Move to GPU
    if Constants.USE_GPU:
        nmt_model = nmt_model.cuda()
        critic = critic.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(nmt_model, flags.pretrain_path, exclude_prefix=None, device=Constants.CURRENT_DEVICE)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=nmt_model,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=nmt_model,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 5. Build scheduler for optimizer if needed
    scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])

    # 6. build moving average

    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=nmt_model.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # Reload from latest checkpoint
    if flags.reload:
        checkpoint_saver.load_latest(model=nmt_model,
                                     optim=optim,
                                     lr_scheduler=scheduler,
                                     collections=model_collections,
                                     ma=ma, device=Constants.CURRENT_DEVICE)

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=nmt_model.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

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

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss = 0.0
    cum_n_words = 0
    correct_tokens = 0

    total_tokens = 0
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
            seqs_x, seqs_y = batch

            batch_size = len(seqs_x)
            cum_n_words += sum(len(s) for s in seqs_y)

            try:
                # Prepare data
                x, y = prepare_data(seqs_x, seqs_y, cuda=Constants.USE_GPU)

                loss, t, c = compute_forward(model=nmt_model,
                                             critic=critic,
                                             seqs_x=x,
                                             seqs_y=y,
                                             eval=False,
                                             normalization=1.0,
                                             norm_by_words=training_configs["norm_by_words"])

                update_cycle -= 1
                correct_tokens += c
                total_tokens += t
                grad_denom += batch_size
                train_loss += loss

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

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    train_loss = dist.all_reduce_py(train_loss)
                    cum_n_words = dist.all_reduce_py(cum_n_words)
                    correct_tokens = dist.all_reduce_py(correct_tokens)
                    total_tokens = dist.all_reduce_py(total_tokens)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                    postfix_str = 'TrainLoss: {:.2f}, ValidLoss(best): {:.2f} ({:.2f}), accuracy: {:.4f}'.format(
                        train_loss, valid_loss,
                        best_valid_loss, correct_tokens / total_tokens)
                    training_progress_bar.set_postfix_str(postfix_str)

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_meter.update(train_loss, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss = 0.0
                train_accuracy = correct_tokens / total_tokens
                correct_tokens = 0
                total_tokens = 0

            else:
                continue

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss", scalar_value=train_loss_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                    summary_writer.add_scalar("accuracy", scalar_value=train_accuracy, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(nmt_model):

                    if ma is not None:
                        nmt_model.load_state_dict(ma.export_ma_params(), strict=False)

                    valid_loss, accuracy = loss_evaluation(model=nmt_model,
                                                           critic=critic,
                                                           valid_iterator=valid_iterator,
                                                           rank=rank,
                                                           world_size=world_size)

                if scheduler is not None and optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=valid_loss)

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()
                best_valid_loss = min_history_loss
                if summary_writer is not None:
                    summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                    summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)
                    summary_writer.add_scalar("accuracy", accuracy, global_step=uidx)

                if best_valid_loss == valid_loss:
                    if rank == 0:
                        # 1. save the best model
                        torch.save(nmt_model.state_dict(), best_model_prefix + ".final")

                        # 2. record all several best models
                        best_model_saver.save(global_step=uidx,
                                              model=nmt_model,
                                              optim=optim,
                                              lr_scheduler=scheduler,
                                              collections=model_collections,
                                              ma=ma)
                    bad_count = 0
                else:
                    bad_count += 1

                if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                    is_early_stop = True
                    WARN("Early Stop!")
                    exit(0)

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

                if not is_early_stop:
                    if rank == 0:
                        checkpoint_saver.save(global_step=uidx,
                                              model=nmt_model,
                                              optim=optim,
                                              lr_scheduler=scheduler,
                                              collections=model_collections,
                                              ma=ma)

        if training_progress_bar is not None:
            training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break
