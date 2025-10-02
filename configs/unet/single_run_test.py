# Single run config without YAPF issues
_base_ = './eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py'

# Monkey patch to disable pretty_text YAPF formatting
import mmengine.config.config as config_module
original_pretty_text = config_module.Config.pretty_text

@property
def patched_pretty_text(self):
    """Return raw text instead of YAPF formatted to avoid parse errors"""
    return str(self._cfg_dict)

config_module.Config.pretty_text = patched_pretty_text

# Override parameters for single test run
optimizer = dict(type='SGD', lr=0.0065, momentum=0.9, weight_decay=0.0005)

model = dict(
    decode_head=dict(
        dropout_ratio=0.15,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.3, class_weight=[1.0, 2.5, 2.5]),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=0.7)
        ]
    )
)

train_dataloader = dict(batch_size=4)

# Override CLAHE in pipelines
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadAnnotations'),
    dict(type='CLAHEGrayscale', clip_limit=1.5, tile_grid_size=(8, 8)),
    dict(type='GrayscalePhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5)),
    dict(type='RandomRotate', degree=10, prob=0.5, pad_val=0, seg_pad_val=0),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadAnnotations'),
    dict(type='CLAHEGrayscale', clip_limit=1.5, tile_grid_size=(8, 8)),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='ResetOriShape'),
    dict(type='PackSegInputs')
]
