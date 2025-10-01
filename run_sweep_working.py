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
        dice_scores = []
        for i in range(3):
            dice_match = re.search(rf'decode\.dice_class_{i}:\s*([\d.]+)', line)
            if dice_match:
                metrics[f'dice_class_{i}'] = float(dice_match.group(1))
                if i > 0:  # Only kidney classes (1 and 2)
                    dice_scores.append(float(dice_match.group(1)))

        # Calculate kidney-only mDice
        if len(dice_scores) == 2:
            metrics['kidney_mDice'] = sum(dice_scores) / 2

    # Validation metrics pattern
    elif "Iter(val)" in line or "mDice" in line:
        # Extract per-class Dice scores from table format
        dice_scores = []

        # For left_kidney (class 1)
        if 'left_kidney' in line:
            dice_match = re.search(r'\|\s*left_kidney\s*\|\s*([\d.]+)', line)
            if dice_match:
                dice_scores.append(float(dice_match.group(1)))

        # For right_kidney (class 2)
        if 'right_kidney' in line:
            dice_match = re.search(r'\|\s*right_kidney\s*\|\s*([\d.]+)', line)
            if dice_match:
                dice_scores.append(float(dice_match.group(1)))

        # Calculate kidney-only mDice for validation
        if dice_scores:
            metrics['val_kidney_mDice'] = sum(dice_scores) / len(dice_scores)

        # Still extract regular mDice for reference
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

    # Build command with fixed best hyperparameters and sweep parameters
    ce_weight = 1.0 - config.dice_weight  # Calculate CE weight from dice weight

    cmd = [
        "python3.10", "tools/train.py",
        "configs/unet/sweep_config_clean.py",
        "--work-dir", f"work_dirs/sweep_{run.id}",
        "--cfg-options",
        "optimizer.lr=0.0077",  # Fixed at best value
        "model.decode_head.dropout_ratio=0.1",  # Fixed at best value
        "train_dataloader.batch_size=4",  # Fixed at best value
        # Sweep dice weights
        f"model.decode_head.loss_decode[0].loss_weight={ce_weight:.1f}",
        f"model.decode_head.loss_decode[1].loss_weight={config.dice_weight:.1f}",
        # Sweep CLAHE parameter (index 2 in train_pipeline confirmed)
        f"train_pipeline[2].clip_limit={config.clahe_clip_limit:.1f}",
        f"val_pipeline[2].clip_limit={config.clahe_clip_limit:.1f}",  # Also update validation pipeline
        "train_cfg.max_iters=4000",
        "train_cfg.val_interval=400",
    ]

    # Set environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '/content/mmsegmentation'
    env['MMENGINE_DISABLE_PRETTY_TEXT'] = '1'  # Disable YAPF formatting to avoid syntax errors

    # Print all configuration details for verification
    print("="*60)
    print("SWEEP CONFIGURATION VERIFICATION")
    print("="*60)
    print(f"Metric optimizing: val_kidney_mDice (avg of class 1 & 2 only)")
    print(f"Fixed parameters:")
    print(f"  - Learning rate: 0.0077")
    print(f"  - Dropout: 0.1")
    print(f"  - Batch size: 4")
    print(f"  - Class weights: [1.0, 3.0, 3.0]")
    print(f"Sweeping parameters:")
    print(f"  - Dice weight: {config.dice_weight:.1f} (CE weight: {ce_weight:.1f})")
    print(f"  - CLAHE clip_limit: {config.clahe_clip_limit:.2f}")
    print("="*60)
    print("Command:", " ".join(cmd))
    print("="*60)

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
        'metric': {'name': 'val_kidney_mDice', 'goal': 'maximize'},  # Focus on kidneys only!
        'parameters': {
            'clahe_clip_limit': {'values': [1.0, 2.0, 4.0]},  # Specific CLAHE values
            'dice_weight': {'values': [0.7, 0.8]}  # Test dice weights
        }
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="cat-kidney-sweep-working")
    print(f"Created sweep with ID: {sweep_id}")

    # Run sweep agent
    wandb.agent(sweep_id, train, count=15)  # 15 trials as originally planned