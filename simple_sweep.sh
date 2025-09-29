#!/bin/bash
# Simple parameter sweep without wandb sweep API
# Run with: bash simple_sweep.sh

# Arrays of hyperparameters to test
LRS=(0.005 0.01 0.02)
CLASS_WEIGHTS=(2.0 2.5 3.0 3.5)
DICE_WEIGHTS=(0.4 0.5 0.6 0.7)
CLAHE_LIMITS=(1.5 2.0 2.5 3.0)

# Create results directory
mkdir -p sweep_results

# Run experiments
for lr in "${LRS[@]}"; do
    for cw in "${CLASS_WEIGHTS[@]}"; do
        for dw in "${DICE_WEIGHTS[@]}"; do
            for clahe in "${CLAHE_LIMITS[@]}"; do
                ce_weight=$(echo "1.0 - $dw" | bc -l)

                exp_name="lr${lr}_cw${cw}_dice${dw}_clahe${clahe}"
                echo "Running experiment: $exp_name"

                python tools/train.py \
                    configs/unet/eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py \
                    --cfg-options \
                    optimizer.lr=$lr \
                    "model.decode_head.loss_decode=[dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=$ce_weight,class_weight=[1.0,$cw,$cw]),dict(type='DiceLoss',use_sigmoid=False,loss_weight=$dw)]" \
                    train_pipeline[2].clip_limit=$clahe \
                    visualizer.vis_backends[1].init_kwargs.name="$exp_name" \
                    train_cfg.max_iters=3000 \
                    train_cfg.val_interval=300 \
                    work_dir="sweep_results/$exp_name"
            done
        done
    done
done