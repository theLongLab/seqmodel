#!/bin/bash

python src/exp/seqbert/pretrain.py \
    --n_dims=256 \
    --n_heads=2 \
    --n_layers=2 \
    --n_decode_layers=2 \
    --feedforward_dims=512 \
    --dropout=0.0 \
    --position_embedding=Sinusoidal \
    --keep_prop=0.03 \
    --mask_prop=0.1 \
    --random_prop=0.02 \
    --cls_regularization=1. \
    --num_workers=4 \
    --batch_size=4 \
    --learning_rate=3e-4 \
    --seq_len=100 \
    --seq_file=data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa \
    --train_intervals=data/ref_genome/grch38-train.bed \
    --valid_intervals=data/ref_genome/grch38-1M-valid.bed \
    --default_root_dir='outputs' \
    --accumulate_grad_batches=8 \
    --print_progress_freq=100 \
    --save_checkpoint_freq=1000 \
    --gpus=0 \
    # --load_checkpoint_path=$OUT_DIR/lightning_logs/version_53753746/checkpoints/N-Step-Checkpoint_0_85000.ckpt \
    # --seq_file=data/ref_genome/test-2k.fa \
    # --DEBUG_use_random_data=False \
    # --DEBUG_random_repeat_len=2 \
    # --DEBUG_random_n_repeats=100 \
