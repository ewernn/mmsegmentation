_base_ = './eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py'

# Override visualizer to remove WandB for sweep compatibility
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],  # Only local, no WandB
    name='visualizer'
)