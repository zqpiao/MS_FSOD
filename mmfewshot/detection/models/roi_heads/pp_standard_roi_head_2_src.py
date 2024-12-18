# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from mmdet.core import bbox2roi, bbox_overlaps
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
from torch import Tensor
from mmdet.core import bbox2result

import  numpy as np
import os
from torchvision.ops import  roi_align

from mmdet.models.roi_heads.bbox_heads import convfc_bbox_head

@HEADS.register_module()
class PP_StandardRoIHead_2(StandardRoIHead):
    """RoI head for `FSCE <https://arxiv.org/abs/2103.05950>`_."""


    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        # print('herhe')
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        # cls_score, bbox_pred = self.bbox_head(bbox_feats)
        # cls_score, bbox_pred, _ = self.bbox_head(bbox_feats)

        cls_score, bbox_pred, x_cls = self.bbox_head(bbox_feats)
        # cls_score, bbox_pred, _ = self.bbox_head(bbox_feats)

        # bbox_results = dict(
        #     cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)


        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=x_cls)
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


        # ####--------保存RoI特征-------
        # det_bboxes_save, det_labels_save = self.simple_test_bboxes(
        #     x, img_metas, proposal_list, self.test_cfg, rescale=False)
        # rois = bbox2roi(det_bboxes_save)
        # bbox_feats = self.roi_extractors(
        #     x[:self.roi_extractors.num_inputs], rois)

        # bbox_feats = self.bbox_roi_extractor(
        #     x[:self.bbox_roi_extractor.num_inputs], rois)

        # cls_score, bbox_pred = self.bbox_head(bbox_feats)

        # cls_score, bbox_pred, roi_featuress = self.bbox_head(bbox_feats)

        #
        # scale = [4, 8, 16, 32, 64]
        # bbox_feats = roi_align(x[4], rois,[7, 7], 1/scale[4])

        # bbox_feats = self.bbox_roi_extractor(
        #     x[-1], rois)


        # save_path_root = r"/data/piaozhengquan/projects/FSOD/mmfewshot-main/work_dirs/tsne_roi_features/base_training"

        # save_path_root = r"/data/piaozhengquan/projects/FSOD/mmfewshot-main/work_dirs/tsne_roi_features/base_training_torch_vision"

        # save_path_root = r"/data/piaozhengquan/projects/FSOD/mmfewshot-main/work_dirs/tsne_roi_features/base_training_novel"

        # save_path_root = r"/data/piaozhengquan/projects/FSOD/mmfewshot-main/work_dirs/tsne_roi_features/base_training_all"
        #
        # if not os.path.exists(save_path_root):
        #     os.mkdir(save_path_root)
        #
        # print('bbox_feats.shape[0]: ', bbox_feats.shape[0])
        # if bbox_feats.shape[0] > 0:
        #     for nn in range(bbox_feats.shape[0]):
        #         tem = bbox_feats[nn]
        #         # print('tem.shape: ', tem.shape)
        #         a_feature = bbox_feats[nn].reshape(-1).detach().cpu().numpy()
        #         label = det_labels_save[0].detach().cpu().numpy()[nn]
        #
        #         # print('det_labels: ', det_labels)
        #         # print(label)
        #
        #         save_path = os.path.join(save_path_root, str(label))
        #
        #         if not os.path.exists(save_path):
        #             os.mkdir(save_path)
        #
        #         num = len(os.listdir(save_path))
        #
        #         # if num > 300:
        #         #     break
        #
        #         np.save(os.path.join(save_path, str(num+1)+'.npy'), a_feature)




        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            # return bbox_results
            # return bbox_results, x, self.bbox_roi_extractor, self.bbox_roi_extractor.num_inputs
            return bbox_results, x, self.bbox_roi_extractor, self.bbox_roi_extractor.num_inputs, self.bbox_head

        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    # def _bbox_forward(self, x: List[Tensor], rois: Tensor) -> Dict:
    #     """Box head forward function used in both training and testing phase.
    #
    #      Args:
    #          x (list[Tensor]): Features from the upstream network,
    #             each is a 4D-tensor.
    #          rois (Tensor): Shape of (num_proposals, 4) or (num_proposals, 5).
    #
    #     Returns:
    #          dict[str, Tensor]: A dictionary of predicted results and output
    #              features.
    #     """
    #     bbox_feats = self.roi_extractors(
    #         x[:self.roi_extractors.num_inputs], rois)
    #
    #
    #     print('heheh')
    #     if self.with_shared_head:
    #         bbox_feats = self.shared_head(bbox_feats)
    #     cls_score, bbox_pred, contrast_feat = self.bbox_head(bbox_feats)
    #     bbox_results = dict(
    #         cls_score=cls_score,
    #         bbox_pred=bbox_pred,
    #         bbox_feats=bbox_feats,
    #         contrast_feat=contrast_feat)
    #     return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

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
            return bbox_results, x, self.bbox_roi_extractor, self.bbox_roi_extractor.num_inputs
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))
