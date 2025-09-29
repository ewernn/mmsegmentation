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
        "configs/unet/sweep_config.py",  # Use sweep-specific config without WandB
        "--cfg-options",
        f"optimizer.lr={config.lr}",
        f"model.decode_head.dropout_ratio={config.dropout}",
        f"train_dataloader.batch_size={config.batch_size}",
        # Skip loss_decode modifications for now - too complex for command line
        # Just vary the simpler parameters
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

    # Run training WITHOUT capturing output so we see it in real-time
    print("Starting training...")
    result = subprocess.run(cmd, env=env)

    # Check if training succeeded
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    else:
        print("Training completed successfully!")
        # Log a final metric to indicate completion
        wandb.log({'training_completed': 1})

if __name__ == "__main__":
    # Simplified sweep focusing on params we can actually modify
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'mDice', 'goal': 'maximize'},
        'parameters': {
            'lr': {'distribution': 'log_uniform_values', 'min': 0.005, 'max': 0.03},
            'dropout': {'values': [0.1, 0.2, 0.3]},
            'batch_size': {'values': [2, 4]},  # Reduced from [4, 8] to avoid OOM
            # Remove complex loss params for now
            'dice_weight': {'values': [0.6]},  # Just use one value
            'class_weight_kidney': {'values': [2.5]}  # Just use one value
        }
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="cat-kidney-sweep-v2")
    print(f"Created sweep with ID: {sweep_id}")

    # Run the sweep agent
    wandb.agent(sweep_id, train, count=15)  # 15 trials for focused search