#!/bin/bash

python src/exp/seqbert/pretrain_cadd.py \
    --n_dims=256 \
    --n_heads=2 \
    --n_layers=2 \
    --n_decode_layers=2 \
    --feedforward_dims=512 \
    --dropout=0.0 \
    --position_embedding=Sinusoidal \
    --num_workers=4 \
    --batch_size=4 \
    --learning_rate=3e-4 \
    --seq_len=100 \
    --seq_file=data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa \
    --vcf_file=data/vcf/ALL.chr22.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz \
    --train_intervals=data/vcf/chr22-1000-seq-10-variants.bed \
    --valid_intervals=data/vcf/chr22-1000-seq-10-variants.bed \
    --default_root_dir=outputs \
    --print_progress_freq=10 \
    --save_checkpoint_freq=100 \
    --val_check_interval=10 \
    --limit_val_batches=10 \
    --gpus=0 \
    # --DEBUG_use_random_data=True \

    # --accumulate_grad_batches=1 \
    # --load_checkpoint_path=$OUT_DIR/lightning_logs/version_53753746/checkpoints/N-Step-Checkpoint_0_85000.ckpt \
    # --seq_file=data/ref_genome/test-2k.fa \
