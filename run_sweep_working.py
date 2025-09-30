#!/usr/bin/env python3
"""
Working WandB Sweep Runner that actually logs metrics
"""
import wandb
import subprocess
import os
import time
import re

def parse_metrics_from_line(line):
    """Extract metrics from training output lines"""
    metrics = {}

    # Training metrics pattern
    if "Iter(train)" in line:
        # Extract iteration number
        iter_match = re.search(r'\[\s*(\d+)/\d+\]', line)
        if iter_match:
            metrics['iteration'] = int(iter_match.group(1))

        # Extract loss values
        loss_match = re.search(r'loss:\s*([\d.]+)', line)
        if loss_match:
            metrics['loss'] = float(loss_match.group(1))

        # Extract learning rate
        lr_match = re.search(r'lr:\s*([\d.e-]+)', line)
        if lr_match:
            metrics['learning_rate'] = float(lr_match.group(1))

        # Extract dice scores
        for i in range(3):
            dice_match = re.search(rf'dice_class_{i}:\s*([\d.]+)', line)
            if dice_match:
                metrics[f'dice_class_{i}'] = float(dice_match.group(1))

    # Validation metrics pattern
    elif "Iter(val)" in line or "mDice" in line:
        # Extract mDice
        dice_match = re.search(r'mDice[:\s]+([\d.]+)', line)
        if dice_match:
            metrics['val_mDice'] = float(dice_match.group(1))

        # Extract mIoU
        iou_match = re.search(r'mIoU[:\s]+([\d.]+)', line)
        if iou_match:
            metrics['val_mIoU'] = float(iou_match.group(1))

    return metrics

def train():
    # Initialize wandb run within sweep
    run = wandb.init()
    config = wandb.config

    # Build command with hyperparameters
    ce_weight = 1.0 - config.dice_weight

    cmd = [
        "python3.10", "tools/train.py",
        "configs/unet/sweep_config.py",
        "--cfg-options",
        f"optimizer.lr={config.lr}",
        f"model.decode_head.dropout_ratio={config.dropout}",
        f"train_dataloader.batch_size={config.batch_size}",
        "train_cfg.max_iters=4000",  # Full training
        "train_cfg.val_interval=400",
        "log_config.interval=50",  # Log every 50 iterations
    ]

    # Set environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '/content/mmsegmentation'

    print(f"Starting training with lr={config.lr:.4f}, dropout={config.dropout}, batch_size={config.batch_size}")

    # Run training with real-time output processing
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1  # Line buffering
    )

    # Read output line by line and parse metrics
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())  # Show output in console

            # Parse and log metrics to wandb
            metrics = parse_metrics_from_line(line)
            if metrics:
                wandb.log(metrics)
                print(f"Logged metrics: {metrics}")

    # Wait for process to complete
    process.wait()

    if process.returncode != 0:
        print(f"Training failed with return code {process.returncode}")
        raise subprocess.CalledProcessError(process.returncode, cmd)
    else:
        print("Training completed successfully!")
        wandb.log({'training_completed': 1})

if __name__ == "__main__":
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_mDice', 'goal': 'maximize'},
        'parameters': {
            'lr': {'distribution': 'uniform', 'min': 0.005, 'max': 0.008},  # Narrowed around 6e-3
            'dropout': {'values': [0.1]},  # Best performer
            'batch_size': {'values': [2, 4]},  # Keep both options
            'dice_weight': {'values': [0.4, 0.6, 0.7]},  # Now exploring!
            'class_weight_kidney': {'values': [1.5, 2.5, 3.5]}  # Now exploring!
        }
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="cat-kidney-sweep-working")
    print(f"Created sweep with ID: {sweep_id}")

    # Run sweep agent
    wandb.agent(sweep_id, train, count=15)  # 15 trials as originally planned