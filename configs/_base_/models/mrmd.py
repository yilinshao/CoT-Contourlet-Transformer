# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=dict(
            vit_pretrained='/Document/SETR/transfer/jx_vit_large_p16_384.pth',
            resnet_pretrained='/Document/SETR/transfer/resnet50.pth'
        ),
    backbone=dict(
        type='VIT_CNN',
        vit_backbone=dict(
            type='VisionTransformer',
            model_name='vit_large_patch16_384',
            img_size=480,
            patch_size=16,
            in_chans=3,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            num_classes=19,
            drop_rate=0.1,
            norm_cfg=norm_cfg,
            pos_embed_interp=True,
            align_corners=False,
        ),
        resnet_backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 1, 1),
            strides=(1, 2, 2, 2),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True
        )
    ),
    decode_head=dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=23,
        img_size=768,
        embed_dim=1024,
        num_classes=19,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
