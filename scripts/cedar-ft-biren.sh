#!/bin/bash
#SBATCH --job-name=seqbert-ft-biren
#SBATCH --account=def-quanlong            # needed for resource billing if using compute canada
#SBATCH --time=2:00:00                           # max walltime in D-HH:MM or HH:MM:SS
#SBATCH --cpus-per-task=4             # number of cores
#SBATCH --gres=gpu:p100:1                 # type and number of GPU(s) per node
#SBATCH --mem=16000                                       # max memory (default unit is MB) per node
#SBATCH --output=seqbert-ft-biren-%j.out                 # file name for the output
#SBATCH --error=seqbert-ft-biren-%j.err                  # file name for errors
                                                          # %j gets replaced by the job number

## project name
NAME_DIR=seqbert-ft-biren
cd ~/proj/$NAME_DIR

## on compute canada, scratch dir is for short term and home dir is for long term
## note: code below assumes scratch has been linked to home directory
## e.g. `ln -s /scratch/dkwok/ ~/scratch`

## load modules
## use `module avail`, `module spider` to find relevant modules
module load nixpkgs/16.09  gcc/7.3.0 cuda/10.1 cudnn/7.6.5 python/3.7.4

## setup virtual environment
## compute canada only uses virtualenv and pip
## do not use conda as conda will write files to home directory
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

## install project dependencies
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.txt
## these dependencies need to be downloaded
pip install pyfaidx pytorch-lightning

## extract all data files in tar and tar.gz formats
## compute canada guidelines say to keep files archived to reduce disk utilization
## when accessing data, use `$SLURM_TMPDIR` which is fastest storage directly attached to compute nodes
mkdir $SLURM_TMPDIR/$NAME_DIR
tar xzf ~/data/$NAME_DIR/*.tar.gz -C $SLURM_TMPDIR/$NAME_DIR

## make output dir if does not exist
mkdir ~/scratch/$NAME_DIR

# hparams
python ./src/exp/seqbert/finetune-biren.py \
    --n_dims=512 \
    --n_heads=4 \
    --n_layers=4 \
    --n_decode_layers=2 \
    --feedforward_dims=1024 \
    --dropout=0.0 \
    --position_embedding=Sinusoidal \
    --batch_size=32 \
    --learning_rate=3e-4 \
    --seq_len=1000 \
    --default_root_dir=~/scratch/$NAME_DIR \
    --print_progress_freq=500 \
    --save_checkpoint_freq=5000 \
    --num_workers=8 \
    --seq_file=$SLURM_TMPDIR/$NAME_DIR/data/vista/all-enhancers.fa \
    --train_intervals=$SLURM_TMPDIR/$NAME_DIR/data/vista/human-enhancers-train.bed \
    --valid_intervals=$SLURM_TMPDIR/$NAME_DIR/data/vista/human-enhancers-valid.bed \
    --test_intervals=$SLURM_TMPDIR/$NAME_DIR/data/vista/human-enhancers-test.bed \
    --seq_len_source_multiplier=2. \
    --crop_factor=0.3 \
    --seq_len_sample_freq=0.25 \
    --load_pretrained_model=./lightning_logs/version_55349874/checkpoints/fixed-N-Step-Checkpoint_0_260000.ckpt \

    # --load_checkpoint_path='' \
    # --accumulate_grad_batches=1 \

## clean up by stopping virtualenv
deactivate
