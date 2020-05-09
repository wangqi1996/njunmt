# coding=utf-8

# teacher choice
import os
import re

import torch

from src.models import build_model
from src.utils import Constants
from src.utils.logging import INFO

AVE_BEST_K = 'ave_best_k'
BEST = 'best'
MA = 'ma'


def get_odc_k(ave_k):
    pattern = re.compile("ave_best_\d+")
    match = pattern.match(ave_k)
    if match is not None:
        k_list = [int(k) for k in ave_k.split('_') if k.isdigit()]
        if len(k_list) > 1:
            raise ValueError(u"ave_best_k error!")
        else:
            return k_list[0]
    else:
        raise ValueError(u"ave_best_k error!")


def check_odc_config(training_configs):
    ave_best_k = 0
    if training_configs['teacher_choice'] == 'ma':
        assert training_configs['moving_average_method'] is not None, \
            "Using MA as teacher need choose one kind of moving-average method!"
    elif "ave_best" in training_configs['teacher_choice']:
        # check whether "teacher_choice" match pattern "ave_best_k"
        ave_best_k = get_odc_k(training_configs['teacher_choice'])
        assert training_configs['num_kept_best_k_checkpoints'] >= ave_best_k, \
            "When using ave_best_k as teacher, we should at least keep k best checkpoints"
    else:
        pass

    return ave_best_k


def get_teacher_model(training_configs, model_configs, vocab_src, vocab_tgt, flags):
    # build teacher model
    if training_configs['use_odc']:
        INFO('Building teacher model...')

        teacher_model = build_model(n_src_vocab=vocab_src.max_n_words,
                                    n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, vocab_src=vocab_src,
                                    **model_configs)
        if Constants.USE_GPU:
            teacher_model.cuda()

        if training_configs.get('teacher_model_path', '') != '':
            teacher_model_path = training_configs['teacher_model_path']
            teacher_model.load_state_dict(
                torch.load(teacher_model_path, map_location=Constants.CURRENT_DEVICE), strict=False)
        else:
            teacher_model_path = os.path.join(flags.saveto, flags.model_name + '.teacher.pth')
        INFO('Done.')
    else:
        teacher_model = None
        teacher_model_path = ''

    return teacher_model, teacher_model_path
