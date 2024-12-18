# Copyright (c) OpenMMLab. All rights reserved.
from .custom_hook import ContrastiveLossDecayHook
from .pp_get_protype_hook import GETPROTOTYPEHook
from .pp_get_protype_hook_multiscale import MULTISCAL_GETPROTOTYPEHook
from .pp_get_protype_hook_novel import  GETNOVELPROTOTYPEHook
from .ema_hook import  PP_EMAHook

__all__ = ['ContrastiveLossDecayHook', 'GETPROTOTYPEHook', 'MULTISCAL_GETPROTOTYPEHook', 'GETNOVELPROTOTYPEHook',
           'PP_EMAHook']
