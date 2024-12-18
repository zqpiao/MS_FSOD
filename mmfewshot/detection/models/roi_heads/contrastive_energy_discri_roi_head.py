# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from mmdet.core import bbox2roi, bbox_overlaps
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
from torch import Tensor
from mmdet.core import bbox2result


@HEADS.register_module()
class DISCRIContrastiveENERGYRoIHead(StandardRoIHead):
    """RoI head for `FSCE <https://arxiv.org/abs/2103.05950>`_."""

    def _bbox_forward(self, x: List[Tensor], rois: Tensor) -> Dict:
        """Box head forward function used in both training and testing phase.

         Args:
             x (list[Tensor]): Features from the upstream network,
                each is a 4D-tensor.
             rois (Tensor): Shape of (num_proposals, 4) or (num_proposals, 5).

        Returns:
             dict[str, Tensor]: A dictionary of predicted results and output
                 features.
        """
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        # cls_score, bbox_pred, contrast_feat = self.bbox_head(bbox_feats)
        cls_score, bbox_pred, contrast_feat, dis_logit = self.bbox_head(bbox_feats)
        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_feats,
            contrast_feat=contrast_feat,
            dis_logit=dis_logit
        )
        return bbox_results

    def _bbox_forward_train(self, x: List[Tensor],
                            sampling_results: List[object],
                            gt_bboxes: List[Tensor], gt_labels: List[Tensor],
                            img_metas: List[Dict]) -> Dict:
        """Forward function and calculate loss for box head in training phase.

        Args:
            x (list[Tensor]): Features from the upstream network,
                each is a 4D-tensor.
            sampling_results (list[obj:`SamplingResult`]): Sampling result.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of output results and losses.
        """
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        proposal_ious = []
        for res in sampling_results:
            single_pos_proposal_ious = bbox_overlaps(
                res.pos_bboxes, res.pos_gt_bboxes, is_aligned=True)
            single_proposal_ious = torch.zeros(res.bboxes.size(0)).to(
                single_pos_proposal_ious.device)
            single_proposal_ious[:res.pos_bboxes.
                                 size(0)] = single_pos_proposal_ious
            proposal_ious.append(single_proposal_ious)
        proposal_ious = torch.cat(proposal_ious, dim=0)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        bbox_results['dis_logit'],
                                        rois,
                                        *bbox_targets)
        loss_contrast = self.bbox_head.loss_contrast(
            bbox_results['contrast_feat'],
            proposal_ious,
            labels=bbox_targets[0])
        loss_bbox.update(loss_contrast)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            # return bbox_results
            if self.training:
                return bbox_results, x, self.bbox_roi_extractor, self.bbox_roi_extractor.num_inputs, self.bbox_head
            else: ##test
                return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
