# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os


import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmfewshot.detection.datasets import (build_dataloader, build_dataset,
                                          get_copy_dataset_type)
from mmfewshot.detection.models import build_detector


from mmdet.core import bbox2roi

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFewShot test (and eval) a model')
    
   
    parser.add_argument('--config',
                        default='configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_5shot-fine-tuning_seed0_separating.py',
                        help='test config file path')

    

    parser.add_argument('--checkpoint',
                        default='work_dirs/fsce_r101_fpn_voc-split1_5shot-fine-tuning_seed0_separating/iter_4000.pth',
                        help='checkpoint file')



    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--eval',
        type=str,
        default='mAP',
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        # default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
        args.cfg_options = args.options
    return args


import numpy as np
def compute_prototypes(data_loader, model_p):
    print('Geting the prototypes!')

    all_roi_features = torch.zeros(len(data_loader.dataset.CLASSES), data_loader.dataset.num_base_shots, 1024).cuda()
    num_bbox_saved_per_cls = np.zeros(len(data_loader.dataset.CLASSES), np.int8)  ###保存当前已保存每类目标的数目
    # model_p.eval()
    for i, data in enumerate(data_loader):
        data['img'] = data['img'].data  ###train dataloader  ##list [0] : (1,3,640,864)
        data['img_metas'] = [data['img_metas']] ##list [0]
        labels = data['gt_labels'].data[0][0]  ##tensors
        # print('labels: ',labels)
        # print(data['img_metas'][0].data[0][0]['filename'])
        with torch.no_grad():
            # forward in `test` mode
            data_test = dict()
            data_test['img_metas'] = data['img_metas']
            data_test['img'] = data['img']
            # result, x, bbox_roi_extractor, roi_extractor_num_inputs, bbox_head = model_p(return_loss=False, rescale=True,
            #                                                                            **data_test)

            result, x, bbox_roi_extractor, roi_extractor_num_inputs, bbox_head = model_p(return_loss=False,
                                                                                         rescale=True,
                                                                                         **data_test)

            rois = bbox2roi(data['gt_bboxes'].data[0])
            # print('GT_boxes: ', data['gt_bboxes'].data[0])
            # print('\n')
            # print('rois: ', rois)
            # bbox_feats = self.roi_extractors(
            #     x[:self.roi_extractors.num_inputs], rois)

            bbox_feats = bbox_roi_extractor(
                x[:roi_extractor_num_inputs], rois.cuda())

            # bbox_feats = bbox_head(bbox_feats)[-1]  ##1024
            bbox_feats = bbox_head(bbox_feats)[-2]

            # print('sum________: ', bbox_feats.sum())

            # print('bbox_feats.shape[0]: ', bbox_feats.shape[0])
            if bbox_feats.shape[0] > 0:
                for nn in range(bbox_feats.shape[0]):
                    # tem = bbox_feats[nn]
                    # print('tem.shape: ', tem.shape)
                    # a_feature = bbox_feats[nn].reshape(-1).detach().cpu().numpy()
                    a_feature = bbox_feats[nn].reshape(-1).detach()
                    label = int(labels[nn].numpy())

                    all_roi_features[label, num_bbox_saved_per_cls[label], :] = a_feature
                    num_bbox_saved_per_cls[label] += 1

    # print('num_bbox_saved_per_cls: ', num_bbox_saved_per_cls)
    prototype_per_cls = all_roi_features.mean(dim=1)  ##对不同实例求均值
    return prototype_per_cls

def main():
    args = parse_args()


    assert args.out or args.eval or args.show \
        or args.show_dir, (
            'Please specify at least one operation (save/eval/show the '
            'results / save the results) with the argument "--out", "--eval"',
            '"--show" or "--show-dir"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # cfg.data.test.pipeline[1].flip = True

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # currently only support single images testing
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    assert samples_per_gpu == 1, 'currently only support single images testing'
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the prototype dataloader
    dataset_proto = build_dataset(cfg.data.train)
    data_loader_proto = build_dataloader(
        dataset_proto,
        samples_per_gpu=1,
        workers_per_gpu=4,
        # dist=distributed,
        shuffle=False)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # pop frozen_parameters
    cfg.model.pop('frozen_parameters', None)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES



    # for meta-learning methods which require support template dataset
    # for model initialization.
    if cfg.data.get('model_init', None) is not None:
        cfg.data.model_init.pop('copy_from_train_dataset')
        model_init_samples_per_gpu = cfg.data.model_init.pop(
            'samples_per_gpu', 1)
        model_init_workers_per_gpu = cfg.data.model_init.pop(
            'workers_per_gpu', 1)
        if cfg.data.model_init.get('ann_cfg', None) is None:
            assert checkpoint['meta'].get('model_init_ann_cfg',
                                          None) is not None
            cfg.data.model_init.type = \
                get_copy_dataset_type(cfg.data.model_init.type)
            cfg.data.model_init.ann_cfg = \
                checkpoint['meta']['model_init_ann_cfg']
        model_init_dataset = build_dataset(cfg.data.model_init)
        # disable dist to make all rank get same data
        model_init_dataloader = build_dataloader(
            model_init_dataset,
            samples_per_gpu=model_init_samples_per_gpu,
            workers_per_gpu=model_init_workers_per_gpu,
            dist=False,
            shuffle=False)

    if not distributed:
        # Please use MMCV >= 1.4.4 for CPU testing!
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        # print('test-prototype_per_cls_gt: ', model.module.roi_head.bbox_head.prototype_per_cls_gt)
        show_kwargs = dict(show_score_thr=args.show_score_thr)
        if cfg.data.get('model_init', None) is not None:
            from mmfewshot.detection.apis import (single_gpu_model_init,
                                                  single_gpu_test)
            single_gpu_model_init(model, model_init_dataloader)
        else:
            from mmdet.apis.test import single_gpu_test

        model.eval()

        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        if cfg.data.get('model_init', None) is not None:
            from mmfewshot.detection.apis import (multi_gpu_model_init,
                                                  multi_gpu_test)
            multi_gpu_model_init(model, model_init_dataloader)
        else:
            from mmdet.apis.test import multi_gpu_test


        outputs = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
        )

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            # print(dataset.evaluate(outputs, **eval_kwargs))
            eval_results = dataset.evaluate(outputs, **eval_kwargs)

            
if __name__ == '__main__':
    main()
