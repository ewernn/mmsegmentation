#!/usr/bin/env python3
"""
WandB Sweep Runner for MMSegmentation
Run with: python run_sweep.py
"""
import wandb
import subprocess
import os

def train():
    # Initialize wandb run within sweep
    run = wandb.init()
    config = wandb.config

    # Build command with hyperparameters from sweep
    ce_weight = 1.0 - config.dice_weight

    cmd = [
        "python", "tools/train.py",
        "configs/unet/eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py",
        "--cfg-options",
        f"optimizer.lr={config.lr}",
        f"model.decode_head.dropout_ratio={config.dropout}",
        f"train_dataloader.batch_size={config.batch_size}",
        # Complex nested configs need special handling
        f"model.decode_head.loss_decode="
        f"[dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight={ce_weight:.2f},"
        f"class_weight=[1.0,{config.class_weight_kidney:.1f},{config.class_weight_kidney:.1f}]),"
        f"dict(type='DiceLoss',use_sigmoid=False,loss_weight={config.dice_weight:.2f})]",
        # Update CLAHE in pipeline
        f"train_pipeline[2].clip_limit={config.clahe_clip_limit}",
        # Update wandb run name
        f"visualizer.vis_backends[1].init_kwargs.name='sweep-lr{config.lr:.4f}-cw{config.class_weight_kidney:.1f}-dice{config.dice_weight:.1f}'",
        # Reduce iterations for sweep
        "train_cfg.max_iters=4000",  # Good balance for sweep
        "train_cfg.val_interval=400",  # Validate 10 times during training
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # Option 1: Create and run sweep
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'mDice', 'goal': 'maximize'},
        'parameters': {
            'lr': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 0.05},
            'class_weight_kidney': {'distribution': 'uniform', 'min': 1.5, 'max': 4.0},
            'dice_weight': {'distribution': 'uniform', 'min': 0.4, 'max': 0.8},
            'clahe_clip_limit': {'distribution': 'uniform', 'min': 1.0, 'max': 4.0},
            'dropout': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
            'batch_size': {'values': [2, 4, 8]}
        }
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="cat-kidney-sweep")  # Optional: separate project
    print(f"Created sweep with ID: {sweep_id}")

    # Run the sweep agent
    wandb.agent(sweep_id, train, count=30)  # 30 trials for thorough exploration