_base_ = 'real.py'
custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
checkpoint_file = None

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
num_stages = 3
conv_kernel_size = 1

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='small',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.3,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
),
    decode_head=dict(
        kernel_generate_head=dict(in_channels=[96, 192, 384, 768])),
    auxiliary_head=dict(in_channels=384))

# modify learning rate following the official implementation of Swin Transformer # noqa
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    })

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=8000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
load_from = 'knet_s3_upernet_convnext-small_woodscape_960_2/iter_80000.pth'