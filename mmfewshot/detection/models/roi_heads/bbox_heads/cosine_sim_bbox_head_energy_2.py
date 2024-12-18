# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import ConvFCBBoxHead
from torch import Tensor

from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
import torch.nn.functional as F
from mmdet.core import multiclass_nms
from mmcv.ops.nms import batched_nms


@HEADS.register_module()
class CosineSimBBoxHeadENERGY_2(ConvFCBBoxHead):
    """BBOxHead for `TFA <https://arxiv.org/abs/2003.06957>`_.

    The code is modified from the official implementation
    https://github.com/ucbdrive/few-shot-object-detection/

    Args:
        scale (int): Scaling factor of `cls_score`. Default: 20.
        learnable_scale (bool): Learnable global scaling factor.
            Default: False.
        eps (float): Constant variable to avoid division by zero.
    """

    def __init__(self,
                 scale: int = 20,
                 learnable_scale: bool = False,
                 base_classes: int = 60,
                 loss_energy_weight: float = 0.01,
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

        self.base_classes = base_classes
        self.loss_energy_weight = loss_energy_weight

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



        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()

        ### energy loss
        loss_energy = torch.zeros(1).cuda()
        if cls_score is not None:
            base_num = self.base_classes
            # energy = F.softmax(cls_score, dim=-1)
            energy = cls_score

            cls_score_base = energy[labels < base_num]  ##0_15
            cls_score_novel = energy[(base_num <= labels) & (labels < self.num_classes)]  ##15-19
            # m_in, m_out = -18, -2
            m_in, m_out = -25, -2
            # delta = -20  ##判断是否为novel类的阈值

            if cls_score_novel.shape[0] > 0:
                base_x = torch.cat((cls_score_novel[:, :base_num], cls_score_novel[:, -1].unsqueeze(1)), dim=1)
                Ec_base = -torch.logsumexp(base_x, dim=1)

                # Ec_base = -torch.logsumexp(cls_score_novel[:, :base_num], dim=1)
                Ec_novel = -torch.logsumexp(cls_score_novel[:, base_num:-1], dim=1)
                # loss_energy_novel = 0.1 * (torch.pow(F.relu(Ec_novel - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_base),2).mean())
                # loss_energy += loss_energy_novel

                # out loss
                # loss_energy +=  0.5 * (-(cls_score_novel[:, :base_num].mean(1) - torch.logsumexp(cls_score_novel[:, :base_num], dim=1)).mean())

                # loss_energy +=  (-(cls_score_novel[:, :base_num].mean(1) - torch.logsumexp(cls_score_novel[:, :base_num], dim=1)).mean() + 0.1 * torch.pow(F.relu(m_out - Ec_base), 2).mean())

                loss_energy += 0.01 * torch.pow(F.relu(m_out - Ec_base), 2).mean()
                loss_energy += 0.01 * torch.pow(F.relu(Ec_novel - m_in), 2).mean()

            if cls_score_base.shape[0] > 0:
                Ec_base = -torch.logsumexp(cls_score_base[:, :base_num], dim=1)
                # Ec_novel = -torch.logsumexp(cls_score_base[:, base_num:-1], dim=1)
                Ec_novel = -torch.logsumexp(cls_score_base[:, base_num:], dim=1)

                # loss_energy_base = 0.1 * (torch.pow(F.relu(Ec_base - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_novel), 2).mean())
                # loss_energy += loss_energy_base

                # loss_energy +=  0.5 * (-(cls_score_base[:, base_num:-1].mean(1) - torch.logsumexp(cls_score_base[:, base_num:-1], dim=1)).mean())

                loss_energy += 0.01 * torch.pow(F.relu(m_out - Ec_novel), 2).mean()
                # loss_energy +=  (-(cls_score_base[:, base_num:-1].mean(1) - torch.logsumexp(cls_score_base[:, base_num:-1], dim=1)).mean() + 0.1 * torch.pow(F.relu(Ec_base - m_in), 2).mean())
                loss_energy += 0.01 * torch.pow(F.relu(Ec_base - m_in), 2).mean()

        losses['loss_energy'] = self.loss_energy_weight * loss_energy

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


        # if cls_score is not None:
        #     avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        #     if cls_score.numel() > 0:
        #         loss_cls_ = self.loss_cls(
        #             cls_score,
        #             labels,
        #             label_weights,
        #             avg_factor=avg_factor,
        #             reduction_override=reduction_override)
        #         if isinstance(loss_cls_, dict):
        #             losses.update(loss_cls_)
        #         else:
        #             losses['loss_cls'] = loss_cls_
        #         if self.custom_activation:
        #             acc_ = self.loss_cls.get_accuracy(cls_score, labels)
        #             losses.update(acc_)
        #         else:
        #             losses['acc'] = accuracy(cls_score, labels)
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
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

            ##根据能量值首先判断是否为base/novel类，然后分别softmax
            # scores = F.softmax(cls_score, dim=-1)
            # m_in, m_out, base_num  = -25, -7, 15
            # Ec_base = -torch.logsumexp(cls_score[:, :base_num], dim=1)
            # if cls_score is not None:
            #     scores[Ec_base <= m_in, :base_num] = F.softmax(
            #         cls_score[Ec_base <= m_in, :base_num], dim=-1)
            #     # scores[Ec_base >= m_out, base_num:] = F.softmax(
            #     #     cls_score[Ec_base >= m_out, base_num:], dim=-1)
            #     scores[Ec_base > m_in, base_num:] = F.softmax(
            #         cls_score[Ec_base > m_in, base_num:], dim=-1)


        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
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
