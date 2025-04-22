# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:09:12 2024

@author: Administrator
"""

_base_ = [
    '../_base_/models/DSASPP_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='repvit_m1_1',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=r'pretrained/repvit_m1_1_distill_300e.pth',
        ),
        out_indices = [3,7,21,24]
    ),
    neck=dict(in_channels=[64, 128, 256, 512]),
    decode_head=dict(num_classes=4))

gpu_multiples = 2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(type='AdamW', lr=0.00015, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric=['mIoU','mFscore'])
