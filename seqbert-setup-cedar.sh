#!/bin/bash

## this script compresses everything in `data/**` and sends to remote cluster
## the code is assumed to be updated via git
TARGET=dkwok@cedar.computecanada.ca
TARGET_DIR=data/seqmodel-seqbert-bp
GZ_FILE=grch38-fasta-and-intervals

tar -czvf $GZ_FILE.tar.gz \
    data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa \
    data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa.fai \
    data/ref_genome/grch38-train.bed \
    data/ref_genome/grch38-1M-valid.bed \
# move to target
ssh $TARGET "mkdir -p $TARGET_DIR"
rsync -r $GZ_FILE.tar.gz $TARGET:$TARGET_DIR/$GZ_FILE.tar.gz
