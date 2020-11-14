#!/bin/bash
#SBATCH --job-name=seqbert-bp
#SBATCH --account=def-quanlong            # needed for resource billing if using compute canada
#SBATCH --time=12:00:00                           # max walltime in D-HH:MM or HH:MM:SS
#SBATCH --cpus-per-task=4             # number of cores
#SBATCH --gres=gpu:p100:1                 # type and number of GPU(s) per node
#SBATCH --mem=16000                                       # max memory (default unit is MB) per node
#SBATCH --output=seqbert-bp%j.out                 # file name for the output
#SBATCH --error=seqbert-bp%j.err                  # file name for errors
                                                          # %j gets replaced by the job number

## project name
NAME_DIR=seqmodel-seqbert-bp

## on compute canada, scratch dir is for short term and home dir is for long term
## note: code below assumes scratch has been linked to home directory
## e.g. `ln -s /scratch/dkwok/ ~/scratch`

## source and data from here
SOURCE_DIR=~/proj/$NAME_DIR
## data archives (archived): use ~/scratch/data for short term, ~/data for long term data
DATA_DIR=~/scratch/$NAME_DIR/data
## outputs to here: use ~/scratch/out for short term, ~/out for long term data
OUT_DIR=~/scratch/$NAME_DIR
## this is for fast disk access
RUN_DIR=$SLURM_TMPDIR

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
pip install --no-index -r $SOURCE_DIR/requirements.txt
## these dependencies need to be downloaded
pip install pyfaidx pytorch-lightning

## extract all data files in tar and tar.gz formats
## compute canada guidelines say to keep files archived to reduce disk utilization
## when accessing data, use `$SLURM_TMPDIR` which is fastest storage directly attached to compute nodes
mkdir $RUN_DIR/$NAME_DIR
tar xf $DATA_DIR/*.tar -C $RUN_DIR/$NAME_DIR
tar xzf $DATA_DIR/*.tar.gz -C $RUN_DIR/$NAME_DIR

## make output dir if does not exist
mkdir $OUT_DIR

# hparams
python $SOURCE_DIR/src/experiment/seqbert.py \
    --deterministic=True \
    --n_dims=512 \
    --n_heads=4 \
    --n_layers=4 \
    --n_decode_layers=4 \
    --feedforward_dims=1024 \
    --position_embedding=Sinusoidal \
    --batch_size=32 \
    --learning_rate=3e-4 \
    --seq_len=1000 \
    --dropout=0.0 \
    --keep_prop=0.03 \
    --mask_prop=0.1 \
    --random_prop=0.02 \
    --cls_regularization=0.01 \
    --num_workers=4 \
    --print_progress_freq=500 \
    --save_checkpoint_freq=5000 \
    --seq_file=$RUN_DIR/$NAME_DIR/data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa \
    --train_intervals=$RUN_DIR/$NAME_DIR/data/ref_genome/grch38-train.bed \
    --valid_intervals=$RUN_DIR/$NAME_DIR/data/ref_genome/grch38-1M-valid.bed \
    --default_root_dir=$OUT_DIR \
    # --accumulate_grad_batches=1 \

## clean up by stopping virtualenv
deactivate
