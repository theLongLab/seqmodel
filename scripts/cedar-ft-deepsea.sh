#!/bin/bash
#SBATCH --job-name=seqbert-ft-deepsea
#SBATCH --account=def-quanlong          # needed for resource billing if using compute canada
#SBATCH --time=2:00:00                  # max walltime in D-HH:MM or HH:MM:SS
#SBATCH --cpus-per-task=4               # number of cores
#SBATCH --gres=gpu:p100:1               # type and number of GPU(s) per node
#SBATCH --mem=16000                     # max memory (default unit is MB) per node
#SBATCH --output=seqbert-ft-deepsea-%j.out  # file name for the output
#SBATCH --error=seqbert-ft-deepsea-%j.err   # file name for errors
                                        # %j gets replaced by the job number

## project name
NAME_DIR=seqbert-ft-deepsea     # task name
SRC_DIR=~/proj/$NAME_DIR        # root dir of src
DATA_DIR=~/data/$NAME_DIR       # load .tar.gz data
RUN_DIR=$SLURM_TMPDIR           # save env and tmp data
OUT_DIR=~/scratch/$NAME_DIR     # save outputs

## make output dir if does not exist
mkdir -p $OUT_DIR
## set working dir to src root for python imports
cd $SRC_DIR

## load modules
module load nixpkgs/16.09  gcc/7.3.0 cuda/10.1 cudnn/7.6.5 python/3.7.4

## setup virtual environment
virtualenv --no-download $RUN_DIR/env
source $SLURM_TMPDIR/env/bin/activate

## install project dependencies
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.txt
## these dependencies need to be downloaded
pip install pyfaidx pytorch-lightning

## extract all .tar.gz data files
tar xzf $DATA_DIR/*.tar.gz -C $RUN_DIR

# hparams
python ./src/exp/seqbert/finetune-deepsea.py \
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
    --default_root_dir=$OUT_DIR \
    --accumulate_grad_batches=1 \
    --print_progress_freq=500 \
    --save_checkpoint_freq=5000 \
    --train_mat=$RUN_DIR/data/deepsea/train.mat \
    --valid_mat=$RUN_DIR/data/deepsea/valid.mat \
    --test_mat=$RUN_DIR/data/deepsea/test.mat \
    --load_pretrained_model=./lightning_logs/version_55349874/checkpoints/fixed-N-Step-Checkpoint_0_260000.ckpt \

    # --accumulate_grad_batches=1 \

## clean up by stopping virtualenv
deactivate
