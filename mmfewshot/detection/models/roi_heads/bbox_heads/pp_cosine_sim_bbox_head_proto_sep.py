# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import ConvFCBBoxHead
from torch import Tensor

from mmcv.runner import force_fp32
import torch.nn.functional as F
from mmdet.core import multiclass_nms
from mmcv.ops.nms import batched_nms

import torch.nn.functional as F
from mmdet.models.losses import accuracy

import numpy as np


class CosineSimilarity(nn.Module):

    # def forward(self, tensor_1, tensor_2):
    #     normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    #     normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    #     # return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
    #     # print('normalized_tensor_1.shape: ', normalized_tensor_1.shape)
    #     # print('normalized_tensor_2.shape: ', normalized_tensor_2.shape)
    #     return torch.mm(normalized_tensor_1,normalized_tensor_2.t())

    def forward(eslf, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1, w2).clamp(min=eps)

@HEADS.register_module()
class PP_CosineSimBBoxHead_PROTO_SEP(ConvFCBBoxHead):
    """BBOxHead for `TFA <https://arxiv.org/abs/2003.06957>`_.

    The code is modified from the official implementation
    https://github.com/ucbdrive/few-shot-object-detection/

    Args:
        scale (int): Scaling factor of `cls_score`. Default: 20.
        learnable_scale (bool): Learnable global scaling factor.
            Default: False.
        eps (float): Constant variable to avoid division by zero.
    """

    # def __init__(self,
    #              scale: int = 20,
    #              learnable_scale: bool = False,
    #              eps: float = 1e-5,
    #              *args,
    #              **kwargs) -> None:

    def __init__(self,
                 scale: int = 20,
                 # prototype_per_cls_gt = [],
                 use_proto_avr_score: bool = True,
                 learnable_scale: bool = False,
                 eps: float = 1e-5,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # override the fc_cls in :obj:`ConvFCBBoxHead`
        if self.with_cls:
            self.fc_cls = nn.Linear(
                self.cls_last_dim, self.num_classes + 1, bias=False)

        # learnable global scaling factor
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1) * scale)
        else:
            self.scale = scale
        self.eps = eps

        # self.prototype_per_cls_gt = prototype_per_cls_gt
        self.prototype_per_cls_gt = []
        self.soft_max = nn.Softmax(dim=1)
        self.use_proto_avr_score = use_proto_avr_score
        self.CosineSimilarity = CosineSimilarity()
        # if use_proto_avr_score:
        #     self.proto_cls_head = nn.Sequential(
        #         nn.Linear(self.fc_out_channels, self.fc_out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(self.fc_out_channels, 256))
        # ###增加一背景类
        # self.cls_score = torch.zeros((1000, self.num_classes + 1), dtype=torch.float).cuda()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            x (Tensor): Shape of (num_proposals, C, H, W).

        Returns:
            tuple:
                cls_score (Tensor): Cls scores, has shape
                    (num_proposals, num_classes).
                bbox_pred (Tensor): Box energies / deltas, has shape
                    (num_proposals, 4).
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if x_cls.dim() > 2:
            x_cls = torch.flatten(x_cls, start_dim=1)


        # ####原始分类头
        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_cls_normalized = x_cls.div(x_norm + self.eps)
        # normalize weight
        with torch.no_grad():
            temp_norm = torch.norm(
                self.fc_cls.weight, p=2,
                dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
            self.fc_cls.weight.div_(temp_norm + self.eps)
        # calculate and scale cls_score
        cls_score = self.scale * self.fc_cls(
            x_cls_normalized) if self.with_cls else None


        # ##prototype
        # if isinstance(self.prototype_per_cls_gt,list):
        #     print('It is a list instance:')
        # else:
        #     print('Not a list')

        prototype_per_cls_gt = self.prototype_per_cls_gt
        softmax_proto_x = []

        # if not self.training:
        #     print('pp_prototype_per_cls_gt: ', prototype_per_cls_gt)
        # dot_x_proto = torch.mm(x_cls, prototype_per_cls_gt.t())

        # ###增加一背景类
        # x_cls = self.proto_cls_head(x_cls)

        # cls_score = torch.zeros((x_cls.shape[0], self.num_classes + 1), dtype=torch.float).cuda()


        # ####CosineSimilarity
        # if not isinstance(prototype_per_cls_gt, list):  ##非空
        #     # prototype_per_cls_gt = prototype_per_cls_gt.t()
        #     cls_scores = self.CosineSimilarity(x_cls, prototype_per_cls_gt)
        #     cls_score[:, :cls_scores.shape[1]] = cls_scores  ##损失函数为交叉熵

        ####Prototype sepration







        # ##L2 dist
        # # cls_score = self.cls_score
        # # cls_score[:,-1] = cls_score[:,-1] - 1e8  ###背景类
        # if not isinstance(prototype_per_cls_gt,list):##非空
        #     cls_scores = -1 * torch.cdist(
        #         x_cls.unsqueeze(0), prototype_per_cls_gt.unsqueeze(0),p=2).squeeze(0)
        #
        #     ###softmax_proto_x = self.soft_max(cls_scores)
        #
        #     cls_score[:, :cls_scores.shape[1]] = cls_scores ##损失函数为交叉熵



            # # prototype_per_cls_gt = torch.ones(20,1024).cuda()
        # if not isinstance(prototype_per_cls_gt,list):##非空
        #     # prototype_per_cls_gt_norm = torch.norm(prototype_per_cls_gt, p=2, dim=1).unsqueeze(1).expand_as(x)
        #     # prototype_per_cls_gt_normalized = prototype_per_cls_gt.div(prototype_per_cls_gt_norm + self.eps)
        #     dot_x_proto = torch.mm(x_cls,prototype_per_cls_gt.t())
        #     # dot_x_proto = x_cls * prototype_per_cls_gt
        #     softmax_proto_x = self.soft_max(dot_x_proto)


        # return cls_score, bbox_pred, x_cls

        # # return cls_score, bbox_pred
        if not self.use_proto_avr_score:
            return cls_score, bbox_pred, x_cls
        else:
            # return cls_score, bbox_pred, x_cls, softmax_proto_x
            return cls_score, bbox_pred, x_cls, prototype_per_cls_gt

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             prototype_per_cls_gt,   ##(num_cls, 1024)
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()

        ####=base与lovel类间prototype远离

        # index = list(range(prototype_per_cls_gt.shape[0]))
        # index_before = torch.tensor(index)
        # np.random.shuffle(index) ###随机扰乱
        # index_shuffle = torch.tensor(index)
        # sel_item = index_shuffle.eq(index_before)
        #
        # prototype_per_cls_gt_shuffle = prototype_per_cls_gt[index_shuffle]
        #
        # prototype_per_cls_gt_shuffle_sel = prototype_per_cls_gt_shuffle[~sel_item]
        # prototype_per_cls_gt_sel = prototype_per_cls_gt[~sel_item]
        # prototype_cosineSimilarity= self.CosineSimilarity(prototype_per_cls_gt_shuffle_sel, prototype_per_cls_gt_sel)

        # prototype_cosineSimilarity_ = torch.cosine_similarity(prototype_per_cls_gt, prototype_per_cls_gt)
        prototype_cosineSimilarity= self.CosineSimilarity(prototype_per_cls_gt, prototype_per_cls_gt)
        diag = torch.diag(prototype_cosineSimilarity)
        a_diag = torch.diag_embed(diag)
        prototype_cosineSimilarity = prototype_cosineSimilarity - a_diag
        losses['prototype_cosineSimilarity'] = prototype_cosineSimilarity.mean()


        # cls_scores = -1 * torch.cdist(
        #         x_cls.unsqueeze(0), prototype_per_cls_gt.unsqueeze(0),p=2).squeeze(0)


        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   # softmax_proto_x,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        # print('cls_score: ', cls_score)

        # scores = cls_score.new_tensor(cls_score)
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:  ##yes  scores: (1000,n_cls+1) 第n_cls+1为背景
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

            # ##忽略背景 L2 dist
            # scores[:,:-1] = F.softmax(
            #     cls_score[:,:-1], dim=-1) if cls_score is not None else None


        # print('scores: ', scores)

        # ###根据roi与prototype检测相似性对scores进行几何平均（加权）
        # alpha = 0.5
        # if self.use_proto_avr_score and not isinstance(softmax_proto_x, list):
        #     scores = alpha * scores + (1-alpha) * softmax_proto_x



        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:  ##yes
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:  ##yes
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep]
