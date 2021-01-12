#!/bin/bash

python src/exp/seqbert/finetune_biren.py \
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
    --accumulate_grad_batches=1 \
    --print_progress_freq=100 \
    --save_checkpoint_freq=1000 \
