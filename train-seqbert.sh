#!/bin/bash

python src/seqbert.py \
    --n_dims=256 \
    --n_heads=2 \
    --n_layers=2 \
    --n_decode_layers=2 \
    --feedforward_dims=512 \
    --dropout=0.1 \
    --position_embedding=Sinusoidal \
    --keep_prop=0.03 \
    --mask_prop=0.1 \
    --random_prop=0.02 \
    --num_workers=4 \
    --epochs=10 \
    --batch_size=32 \
    --learning_rate=3e-4 \
    --seq_file=data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa \
    --train_intervals=data/ref_genome/grch38-train.bed \
    --valid_intervals=data/ref_genome/grch38-1M-valid.bed \
    --seq_len=500 \
