#!/usr/bin/env python3
"""
Simplified WandB Sweep Runner for MMSegmentation
Focuses on the most important hyperparameters
"""
import wandb
import subprocess
import os

def train():
    # Initialize wandb run within sweep
    run = wandb.init()
    config = wandb.config

    # Build command with core hyperparameters
    ce_weight = 1.0 - config.dice_weight

    cmd = [
        "python3.10", "tools/train.py",
        "configs/unet/eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py",
        "--cfg-options",
        f"optimizer.lr={config.lr}",
        f"model.decode_head.dropout_ratio={config.dropout}",
        f"train_dataloader.batch_size={config.batch_size}",
        # Simplified loss config - just update the weights
        f"model.decode_head.loss_decode[0].loss_weight={ce_weight:.2f}",
        f"model.decode_head.loss_decode[0].class_weight=[1.0,{config.class_weight_kidney:.1f},{config.class_weight_kidney:.1f}]",
        f"model.decode_head.loss_decode[1].loss_weight={config.dice_weight:.2f}",
        # Simplified training config
        "train_cfg.max_iters=4000",
        "train_cfg.val_interval=400",
    ]

    # Set environment with PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = '/content/mmsegmentation'

    print(f"Running command: {' '.join(cmd)}")

    # Log the hyperparameters to wandb
    wandb.config.update({
        "ce_weight": ce_weight,
        "actual_command": " ".join(cmd)
    })

    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    # Simplified sweep focusing on most impactful params
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'mDice', 'goal': 'maximize'},
        'parameters': {
            'lr': {'distribution': 'log_uniform_values', 'min': 0.005, 'max': 0.03},
            'class_weight_kidney': {'values': [2.0, 2.5, 3.0, 3.5]},
            'dice_weight': {'values': [0.5, 0.6, 0.7]},
            'dropout': {'values': [0.1, 0.2, 0.3]},
            'batch_size': {'values': [4, 8]}
        }
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="cat-kidney-sweep-simple")
    print(f"Created sweep with ID: {sweep_id}")

    # Run the sweep agent
    wandb.agent(sweep_id, train, count=15)  # 15 trials for focused search