img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                   (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                   (1333, 736), (1333, 768), (1333, 800)],
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = 'data/VOCdevkit/'
data = dict(
    # samples_per_gpu=2,
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='FewShotVOCDefaultDataset',
        save_dataset=True,
        ann_cfg=[dict(method='FSCE', setting='SPLIT3_SEED0_2SHOT')],
        img_prefix='data/VOCdevkit/',
        num_novel_shots=2,
        num_base_shots=2,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                           (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                           (1333, 736), (1333, 768), (1333, 800)],
                keep_ratio=True,
                multiscale_mode='value'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes='ALL_CLASSES_SPLIT3',
        use_difficult=False,
        instance_wise=False),
    val=dict(
        type='FewShotVOCDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt')
        ],
        img_prefix='data/VOCdevkit/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes='ALL_CLASSES_SPLIT3'
        # classes='NOVEL_CLASSES_SPLIT1'
    ),
    test=dict(
        type='FewShotVOCDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt')
        ],
        img_prefix='data/VOCdevkit/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        test_mode=True,
        classes='ALL_CLASSES_SPLIT3'))
# evaluation = dict(
#     interval=4500,
#     metric='mAP',
#     class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=0.001,
    step=3000,
    gamma=0.5)

# runner = dict(type='IterBasedRunner', max_iters=50)
model = dict(
    type='FSCE',
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        init_cfg=[
            dict(
                type='Caffe2Xavier',
                override=dict(type='Caffe2Xavier', name='lateral_convs')),
            dict(
                type='Caffe2Xavier',
                override=dict(type='Caffe2Xavier', name='fpn_convs'))
        ]),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        # type='StandardRoIHead',
        type='DISCRIMINATERoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            # type='CosineSimBBoxHead',
            type='DISCRICosineSimBBoxHeadENERGY_2',
            in_channels=256,
            fc_out_channels=1024,

            dis_head_channels=128,
            base_classes=15,
            # loss_energy_weight=0.01,
            loss_energy_weight=0.01,
            loss_discri_weight=0.01,
            # loss_discri_weight=0.0,

            roi_feat_size=7,
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),

            # loss_cls=dict(
            #     type='', use_sigmoid=False, loss_weight=1.0),

            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            init_cfg=[
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='shared_fcs')),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.01)),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_reg', std=0.001))
            ],
            num_shared_fcs=2,
            scale=20)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.4,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)),
    # frozen_parameters=[
    #     'backbone', 'neck', 'rpn_head', 'roi_head.bbox_head.shared_fcs.0'
    # ]
    frozen_parameters=[
        'backbone', 'neck', 'rpn_head'
    ]
)
checkpoint_config = dict(interval=5000)
# checkpoint_config = dict(interval=8)   ##epoch
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
# log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])  ##epoch
# log_config = dict(interval=2*data['train']['num_novel_shots'], hooks=[dict(type='TextLoggerHook')])  ##epoch

custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = 'work_dirs/fsce_r101_fpn_voc-split2_2shot-fine-tuning_seed0_mixing/epoch_50.pth'
load_from = 'work_dirs/fsce_r101_fpn_voc-split3_2shot-fine-tuning_seed0_mixing/iter_2000.pth'


resume_from = None
workflow = [('train', 1)]
use_infinite_sampler = True
seed = 42
# runner = dict(type='EpochBasedRunner', max_epochs=300)

runner = dict(type='IterBasedRunner', max_iters=8000)

evaluation = dict(
    interval=1000,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])

# evaluation = dict(
#     interval=4,
#     metric='mAP',  ##voc
#     class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'],
# )

# evaluation = dict(
#     interval=4,
#     metric='mAP',  ##voc
#     save_best='mAP',
#     class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'],
# )




