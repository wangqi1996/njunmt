# Copyright 2017 Natural Language Processing Group, Nanjing University, zhengzx.142857@gmail.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from src.models.bert import Bert
from src.models.lm import LM
from src.models.transformer_char import Transformer_Char
from src.models.transformer_relative import Transformer_Relative
from src.utils.logging import INFO
from . import CNN
from .CNN import FConvModel
from .dl4mt import *
from .lightconv import LightConvModel
from .lstm import LSTMModel
from .transformer import *

__all__ = [
    "build_model",
    "load_predefined_configs"
]

MODEL_CLS = {
    "Transformer": Transformer,
    "DL4MT": DL4MT,
    "Transformer_Char": Transformer_Char,
    "Bert": Bert,
    "LM": LM,
    "Transformer_Relative": Transformer_Relative,
    "DynamicCNN": LightConvModel,
    "LSTM": LSTMModel,
    "CNN": FConvModel
}

DEFAULT_CONFIGS = {
    "transformer_base": transformer_base_v2,
    "transformer_base_v1": transformer_base_v1,
    "transformer_base_v2": transformer_base_v2,
    "transformer_low_resource": transformer_low_resource,
    "dl4mt_base": dl4mt_base,
    "bert": transformer_base_v2,
}


def build_model(model: str, **kwargs):
    if model not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))
    INFO("model_name:  " + model)
    return MODEL_CLS[model](**kwargs)


def load_predefined_configs(configs: dict, name: str):
    if name not in DEFAULT_CONFIGS:
        return configs
    else:
        configs = DEFAULT_CONFIGS[name](configs)
        return configs
