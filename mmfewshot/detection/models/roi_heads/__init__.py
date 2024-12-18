# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_heads import (ContrastiveBBoxHead, CosineSimBBoxHead,
                         MultiRelationBBoxHead)
from .contrastive_roi_head import ContrastiveRoIHead
from .fsdetview_roi_head import FSDetViewRoIHead
from .meta_rcnn_roi_head import MetaRCNNRoIHead
from .multi_relation_roi_head import MultiRelationRoIHead
from .shared_heads import MetaRCNNResLayer
from .two_branch_roi_head import TwoBranchRoIHead


# from .pp_standard_roi_head_2 import  PP_StandardRoIHead_2
from .pp_standard_roi_head_2_src import  PP_StandardRoIHead_2
from .roi_extractors import PP_SingleRoIExtractor
from .pp_standard_roi_head_cossim_proto_avg import PP_StandardRoIHead_COSSIM_PRPTP_AVG
from .pp_standard_roi_head_2_proto_sep import PP_StandardRoIHead_PROTO_SEP
from .contrastive_energy_roi_head import ContrastiveENERGYRoIHead
from .discriminate_roi_head import DISCRIMINATERoIHead
from .contrastive_energy_discri_roi_head import DISCRIContrastiveENERGYRoIHead


# __all__ = [
#     'CosineSimBBoxHead', 'ContrastiveBBoxHead', 'MultiRelationBBoxHead',
#     'ContrastiveRoIHead', 'MultiRelationRoIHead', 'FSDetViewRoIHead',
#     'MetaRCNNRoIHead', 'MetaRCNNResLayer', 'TwoBranchRoIHead'
# ]

__all__ = [
    'CosineSimBBoxHead', 'ContrastiveBBoxHead', 'MultiRelationBBoxHead',
    'ContrastiveRoIHead', 'MultiRelationRoIHead', 'FSDetViewRoIHead',
    'MetaRCNNRoIHead', 'MetaRCNNResLayer', 'TwoBranchRoIHead', 'PP_SingleRoIExtractor', 'PP_StandardRoIHead_COSSIM_PRPTP_AVG',
    'PP_StandardRoIHead_PROTO_SEP', 'ContrastiveENERGYRoIHead', 'DISCRIMINATERoIHead', 'DISCRIContrastiveENERGYRoIHead'
]
