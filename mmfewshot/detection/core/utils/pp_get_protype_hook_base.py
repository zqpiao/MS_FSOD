# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmcv.runner import HOOKS, Hook, Runner
import torch
import numpy as np
from mmdet.core import bbox2roi

from mmfewshot.detection.datasets import build_dataloader

from tqdm import tqdm

@HOOKS.register_module()
class GETPROTOTYPEHook(Hook):
    """Hook for contrast loss weight decay used in FSCE.

    Args:
        decay_steps (list[int] | tuple[int]): Each item in the list is
            the step to decay the loss weight.
        decay_rate (float): Decay rate. Default: 0.5.
    """

    # def __init__(self,
    #              dataset) -> None:
    #
    #     self.dataset = dataset



    # def get_prototype(self,runner, dataset_loader):
    def _get_prototype(self, runner):

        dataset = runner.data_loader.dataset
        dataset.pipeline.transforms[2].img_scale = [(1333, 800)]
        dataset.pipeline.transforms[3].flip_ratio = 0.0
        model = runner.model
        # data_loader_ = runner.data_loader

        # dataset = build_dataset(self.dataset)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=4,
            # dist=distributed,
            shuffle=False)



        # all_roi_features = torch.zeros(len(data_loader.dataset.CLASSES), data_loader.dataset.num_base_shots,
        #                                1024).cuda()


        ##base
        all_roi_features = torch.zeros(len(data_loader.dataset.CLASSES), 800,
                                       1024).cuda()

        # all_roi_features = torch.zeros(len(data_loader.dataset.CLASSES), data_loader.dataset.num_base_shots,
        #                                256).cuda()

        num_bbox_saved_per_cls = np.zeros(len(data_loader.dataset.CLASSES), np.int8)  ###保存当前已保存每类目标的数目
        # print('model is train: ', model.training)
        # model.eval()
        # print('model is train: ', model.training)
        # print('len(data_loader): ', len(data_loader))
        for i, data in tqdm(enumerate(data_loader)):  ###考虑顺序打乱问题？
        # for i, data_batch in enumerate(data_loader):
        #     if i == (len(data_loader) // data_loader.batch_size) + 2:
        #         break
            # for a_batch in data['img'].data[0]:
            data['img'] = data['img'].data  ###train dataloader
            data['img_metas'] = [data['img_metas']]
            labels = torch.cat(data['gt_labels'].data[0]) ##batch内的进行合并
            # print('labels: ', labels)
            # labels_list = data['gt_labels'].data[0][0]
            # print('labels_list: ', labels_list)
            with torch.no_grad():
                # forward in `test` mode
                data_test = dict()
                data_test['img_metas'] = data['img_metas']
                data_test['img'] = data['img']
                # result, x, bbox_roi_extractor, roi_extractor_num_inputs, bbox_head = model_p(return_loss=False, rescale=True,
                #                                                                            **data_test)

                result, x, bbox_roi_extractor, roi_extractor_num_inputs, bbox_head = model(return_loss=False,
                                                                                             rescale=True,
                                                                                             **data_test)

                rois = bbox2roi(data['gt_bboxes'].data[0])
                # bbox_feats = self.roi_extractors(
                #     x[:self.roi_extractors.num_inputs], rois)

                bbox_feats = bbox_roi_extractor(
                    x[:roi_extractor_num_inputs], rois.cuda())

                bbox_feats = bbox_head(bbox_feats)[-2]  ##1024

                # print('sum________: ', bbox_feats.sum())

                # print('bbox_feats.shape[0]: ', bbox_feats.shape[0])
                if bbox_feats.shape[0] > 0:
                    for nn in range(bbox_feats.shape[0]):
                        # tem = bbox_feats[nn]
                        # print('tem.shape: ', tem.shape)
                        # a_feature = bbox_feats[nn].reshape(-1).detach().cpu().numpy()
                        a_feature = bbox_feats[nn].reshape(-1).detach()
                        label = int(labels[nn].numpy())

                        # print('label: ', label)
                        # print('num_bbox_saved_per_cls[label]: ', num_bbox_saved_per_cls[label])
                        all_roi_features[label, num_bbox_saved_per_cls[label], :] = a_feature

                        num_bbox_saved_per_cls[label] += 1

        # prototype_per_cls = all_roi_features.mean(dim=1)  ##对不同实例求均值

        prototype_per_cls = torch.zeros(len(data_loader.dataset.CLASSES), 1024).cuda()

        for a_cls in range(len(data_loader.dataset.CLASSES)):
            prototype_per_cls[a_cls] =  all_roi_features[a_cls, :num_bbox_saved_per_cls[a_cls]].mean()

        # print('prototype_per_cls_before: ', model.module.roi_head.bbox_head.prototype_per_cls_gt)
        model.module.roi_head.bbox_head.prototype_per_cls_gt = prototype_per_cls  ##更新prototype
        # print('prototype_per_cls_after: ', model.module.roi_head.bbox_head.prototype_per_cls_gt)
        model.train()
        # print('model is train: ', model.training)
        # print('num_bbox_saved_per_cls: ', num_bbox_saved_per_cls)
        # print('prototype_per_cls.shape: ', prototype_per_cls.shape)
        print('PP_GETPROTOTYPEHook')

    def before_train_epoch(self, runner: Runner) -> None:
        self._get_prototype(runner)

    # def before_iter(self, runner: Runner) -> None:
    #     self._get_prototype(runner)

        # runner_iter = runner.iter + 1
        # decay_rate = 1.0
        # # update decay rate by number of iteration
        # for step in self.decay_steps:
        #     if runner_iter > step:
        #         decay_rate *= self.decay_rate
        # # set decay rate in the bbox_head
        # if is_module_wrapper(runner.model):
        #     runner.model.module.roi_head.bbox_head.set_decay_rate(decay_rate)
        # else:
        #     runner.model.roi_head.bbox_head.set_decay_rate(decay_rate)
