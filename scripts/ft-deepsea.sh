#!/bin/bash

python src/exp/seqbert/finetune_deepsea.py \
    --mode='train' \
    --n_dims=128 \
    --n_heads=2 \
    --n_layers=2 \
    --n_decode_layers=2 \
    --feedforward_dims=256 \
    --dropout=0.0 \
    --position_embedding=Sinusoidal \
    --batch_size=16 \
    --learning_rate=3e-4 \
    --seq_len=1000 \
    --default_root_dir='outputs' \
    --print_progress_freq=100 \
    --save_checkpoint_freq=1000 \
    --train_mat='data/deepsea/train.mat' \
    --gpus=0 \
    --limit_test_batches=10 \
    --single_target=1 \

    # --load_pretrained_model=./outputs/lightning_logs/version_59124619/checkpoints/N-Step-Checkpoint_0_0.ckpt \
    # --valid_mat='data/deepsea/valid.mat' \
    # --test_mat='data/deepsea/test.mat' \
    # --accumulate_grad_batches=1 \
