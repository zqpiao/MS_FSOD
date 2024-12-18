# Copyright (c) OpenMMLab. All rights reserved.
# from .proto_supervised_contrastive_loss import SupervisedContrastiveLoss
from .supervised_contrastive_loss import SupervisedContrastiveLoss
from .margin_cross_entropy_loss import MARGIN_CrossEntropyLoss
from .just_novel_margin_cross_entropy_loss import NOVEL_MARGIN_CrossEntropyLoss

__all__ = ['SupervisedContrastiveLoss', 'MARGIN_CrossEntropyLoss', 'NOVEL_MARGIN_CrossEntropyLoss']
