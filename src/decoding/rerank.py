# coding=utf-8

class _Rerank(object):
    """ """

    def compute_score(self):
        raise NotImplementedError


class T2S_Rerank(_Rerank):

    def __init__(self, model_path, model_name):
        # load model and model parameters
        self.model_name = model_name
        self.model_path = model_path
        self.model = None

    def compute_score(self):
        return 0


class R2L_rerank(_Rerank):
    def __init__(self, model_path, model_name):
        # load model and model parameters
        self.model_name = model_name
        self.model_path = model_path
        self.model = None

    def compute_score(self):
        return 0

class LM_rerank(_Rerank):
    def __init__(self, model_path, model_name):
        # load model and model parameters
        self.model_name = model_name
        self.model_path = model_path
        self.model = None

    def compute_score(self):
        return 0

