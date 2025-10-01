_base_ = './eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py'

# Override custom_imports to avoid YAPF parsing errors during sweep
custom_imports = dict(imports=[], allow_failed_imports=False)

# Disable visualizer completely to avoid YAPF issues
vis_backends = [dict(type='LocalVisBackend')]
visualizer = None

# Fixed class weights, but loss weights will be overridden by sweep
model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.3, class_weight=[1.0, 3.0, 3.0]),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=0.7)
        ]
    )
)