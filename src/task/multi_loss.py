import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import src.distributed as dist
from scripts.checkpoints_average import average_checkpoints
from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.vocabulary import Vocabulary
from src.metric.bleu_scorer import SacreBLEUScorer
from src.models import build_model
from src.modules.criterions import NMTCriterion, CombinationCriterion
from src.optim import Optimizer
from src.optim.lr_scheduler import build_scheduler
from src.task.nmt import prepare_configs, set_seed, prepare_data, load_pretrained_model, \
    bleu_evaluation
from src.utils.common_utils import *
from src.utils.common_utils import AverageMeterDict, BestKSaver
from src.utils.configs import pretty_configs
from src.utils.logging import *
from src.utils.moving_average import build_ma
from src.utils.odc_util import check_odc_config, get_teacher_model


def add_dict_value(dict1, dict2):
    """ add dict2 value to dict1"""
    for key, value in dict2.items():
        dict1[key] = dict1.get(key, 0) + value
    return dict1


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

    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs = model(seqs_x, y_inp)
            loss, loss_dict = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization,
                                     seqs_x=seqs_x, y_inp=y_inp)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
                loss_dict = {name: value.div(words_norm).sum().item() for name, value in loss_dict.items()}
            else:
                loss = loss.sum()
                loss_dict = {name: value.sum().item() for name, value in loss_dict.items()}
        torch.autograd.backward(loss)
        return loss.item(), loss_dict
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs = model(seqs_x, y_inp)
            loss, loss_dict = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True,
                                     seqs_x=seqs_x, y_inp=y_inp)
        return loss.item(), {name: value.mean().item() for name, value in loss_dict.items()}


def loss_evaluation(model, critic, valid_iterator, rank=0, world_size=1):
    """
    :type model: Transformer

    :type critic: NMTCriterion

    :type valid_iterator: DataIterator
    """

    n_sents = 0

    sum_loss = 0.0
    sum_loss_dict = dict()
    valid_iter = valid_iterator.build_generator()

    for batch in valid_iter:
        _, seqs_x, seqs_y = batch

        n_sents += len(seqs_x)

        x, y = prepare_data(seqs_x, seqs_y, cuda=Constants.USE_GPU)

        loss, loss_dict = compute_forward(model=model,
                                          critic=critic,
                                          seqs_x=x,
                                          seqs_y=y,
                                          eval=True)

        if np.isnan(loss):
            WARN("NaN detected!")

        sum_loss += float(loss)
        loss_dict = {key: float(value) for key, value in loss_dict.items()}
        sum_loss_dict = add_dict_value(sum_loss_dict, loss_dict)

    if world_size > 1:
        sum_loss = dist.all_reduce_py(sum_loss)
        sum_loss_dict = dist.all_reduce_py(sum_loss_dict)
        n_sents = dist.all_reduce_py(n_sents)

    return float(sum_loss / n_sents), {key: value / n_sents for key, value in sum_loss_dict.items()}


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

    # use odc
    if training_configs['use_odc'] is True:
        ave_best_k = check_odc_config(training_configs)
    else:
        ave_best_k = 0

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

    bleu_scorer = SacreBLEUScorer(reference_path=data_configs["bleu_valid_reference"],
                                  num_refs=data_configs["num_refs"],
                                  lang_pair=data_configs["lang_pair"],
                                  sacrebleu_args=training_configs["bleu_valid_configs"]['sacrebleu_args'],
                                  postprocess=training_configs["bleu_valid_configs"]['postprocess']
                                  )

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

    best_k_saver = BestKSaver(save_prefix="{0}.best_k_ckpt".format(os.path.join(flags.saveto, flags.model_name)),
                              num_max_keeping=training_configs['num_kept_best_k_checkpoints'])

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, vocab_src=vocab_src,
                            **model_configs)
    INFO(nmt_model)

    # build teacher model
    teacher_model, teacher_model_path = get_teacher_model(training_configs, model_configs, vocab_src, vocab_tgt, flags)

    # build critic
    critic = CombinationCriterion(model_configs['loss_configs'], padding_idx=vocab_tgt.pad, teacher=teacher_model)
    # INFO(critic)
    critic.INFO()

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
    ma = build_ma(training_configs, nmt_model.named_parameters())

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
    teacher_patience = model_collections.get_collection("teacher_patience", [training_configs['teacher_patience']])[-1]

    train_loss_meter = AverageMeter()
    train_loss_dict_meter = AverageMeterDict(critic.get_critic_name())
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss = 0.0
    cum_n_words = 0
    train_loss_dict = dict()
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

                loss, loss_dict = compute_forward(model=nmt_model,
                                                  critic=critic,
                                                  seqs_x=x,
                                                  seqs_y=y,
                                                  eval=False,
                                                  normalization=1.0,
                                                  norm_by_words=training_configs["norm_by_words"])

                update_cycle -= 1
                grad_denom += batch_size
                train_loss += loss
                train_loss_dict = add_dict_value(train_loss_dict, loss_dict)


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
                    train_loss_dict = dist.all_reduce_py(train_loss_dict)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                    postfix_str = 'TrainLoss: {:.2f}, ValidLoss(best): {:.2f} ({:.2f}), '.format(train_loss, valid_loss,
                                                                                                 best_valid_loss)
                    for critic_name, loss_value in train_loss_dict.items():
                        postfix_str += (critic_name + ': {:.2f}, ').format(loss_value)
                    training_progress_bar.set_postfix_str(postfix_str)

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_meter.update(train_loss, grad_denom)
                train_loss_dict_meter.update(train_loss_dict, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss = 0.0
                train_loss_dict = dict()

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
                    # add loss for every critic
                    if flags.display_loss_detail:
                        combination_loss = train_loss_dict_meter.value
                        for key, value in combination_loss.items():
                            summary_writer.add_scalar(key, scalar_value=value, global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_meter.reset()
                train_loss_dict_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(nmt_model):

                    valid_loss, valid_loss_dict = loss_evaluation(model=nmt_model,
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

            # ================================================================================== #
            # BLEU Validation & Early Stop
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx,
                                       every_n_step=training_configs['bleu_valid_freq'],
                                       min_step=training_configs['bleu_valid_warmup'],
                                       debug=flags.debug):

                with cache_parameters(nmt_model):

                    valid_bleu = bleu_evaluation(uidx=uidx,
                                                 valid_iterator=valid_iterator,
                                                 batch_size=training_configs["bleu_valid_batch_size"],
                                                 model=nmt_model,
                                                 bleu_scorer=bleu_scorer,
                                                 vocab_src=vocab_src,
                                                 vocab_tgt=vocab_tgt,
                                                 valid_dir=flags.valid_path,
                                                 max_steps=training_configs["bleu_valid_configs"]["max_steps"],
                                                 beam_size=training_configs["bleu_valid_configs"]["beam_size"],
                                                 alpha=training_configs["bleu_valid_configs"]["alpha"],
                                                 world_size=world_size,
                                                 rank=rank,
                                                 )

                model_collections.add_to_collection(key="history_bleus", value=valid_bleu)

                best_valid_bleu = float(np.array(model_collections.get_collection("history_bleus")).max())

                if summary_writer is not None:
                    summary_writer.add_scalar("bleu", valid_bleu, uidx)
                    summary_writer.add_scalar("best_bleu", best_valid_bleu, uidx)

                # If model get new best valid bleu score
                if valid_bleu >= best_valid_bleu:
                    bad_count = 0

                    if is_early_stop is False:
                        if rank == 0:
                            # 1. save the best model
                            torch.save(nmt_model.state_dict(), best_model_prefix + ".final")

                else:
                    bad_count += 1

                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")
                        exit(0)

                if rank == 0:
                    best_k_saver.save(global_step=uidx,
                                      metric=valid_bleu,
                                      model=nmt_model,
                                      optim=optim,
                                      lr_scheduler=scheduler,
                                      collections=model_collections,
                                      ma=ma)

                # ODC
                if training_configs['use_odc'] is True:
                    if valid_bleu >= best_valid_bleu:
                        pass

                        # choose method to generate teachers from checkpoints
                        # - best
                        # - ave_k_best
                        # - ma

                        if training_configs['teacher_choice'] == 'ma':
                            teacher_params = ma.export_ma_params()
                        elif training_configs['teacher_choice'] == 'best':
                            teacher_params = nmt_model.state_dict()
                        elif "ave_best" in training_configs['teacher_choice']:
                            if best_k_saver.num_saved >= ave_best_k:
                                teacher_params = average_checkpoints(best_k_saver.get_all_ckpt_path()[-ave_best_k:])
                            else:
                                teacher_params = nmt_model.state_dict()
                        else:
                            raise ValueError(
                                "can not support teacher choice %s" % training_configs['teacher_choice'])
                        torch.save(teacher_params, teacher_model_path)
                        del teacher_params
                        teacher_patience = 0
                        critic.set_use_KD(False)
                    else:
                        teacher_patience += 1
                        if teacher_patience >= training_configs['teacher_refresh_warmup']:
                            teacher_params = torch.load(teacher_model_path, map_location=Constants.CURRENT_DEVICE)
                            teacher_model.load_state_dict(teacher_params, strict=False)
                            del teacher_params
                            critic.reset_teacher(teacher_model)
                            critic.set_use_KD(True)

                if summary_writer is not None:
                    summary_writer.add_scalar("bad_count", bad_count, uidx)

                info_str = "{0} Loss: {1:.2f} BLEU: {2:.2f} lrate: {3:6f} patience: {4} ".format(
                    uidx, valid_loss, valid_bleu, lrate, bad_count
                )
                for key, value in valid_loss_dict.items():
                    info_str += (key + ': {0:.2f} '.format(value))
                INFO(info_str)

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)
                model_collections.add_to_collection("teacher_patience", teacher_patience)
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
