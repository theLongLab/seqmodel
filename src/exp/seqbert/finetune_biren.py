import sys
sys.path.append('./src')
from argparse import ArgumentParser
import re
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl

from seqmodel.functional import bioseq_to_index, random_crop, random_seq_fill
from seqmodel.functional.log import roc_auc
from seqmodel.seqdata.iterseq import StridedSequence, FastaFile, bed_from_file
from exp.seqbert import TOKENS_BP_IDX
from exp.seqbert.model import SeqBERT, SeqBERTLightningModule, \
            CheckpointEveryNSteps, BinaryPredictTensorMetric, main
from exp.seqbert.pretrain import Pretrain


class LabelRadomizer():

    def __init__(self, randomize_prop, precision=2):
        self.randomize_prop = randomize_prop
        self.precision = 10**precision

    @staticmethod
    def parse_vista_label(str_label):
        is_positive = ('positive' in str_label)  # positive or negative
        is_human = ('Human' in str_label)  # human or mouse
        matches = re.search('(chr[^:]+):(\d+)-', str_label)  # e.g. chr17:35447270-35448478
        seqname = matches.group(1)
        chr_coord_start = int(matches.group(2))  # add true coordinate from label into coord
        return is_positive, is_human, seqname, chr_coord_start

    def transform(self, key, coord):
        is_positive, is_human, seqname, coord_start = self.parse_vista_label(key)
        random_int = hash(seqname + str(coord_start))
        if (random_int % self.precision) < (self.randomize_prop * self.precision):
            is_positive = (0 == ((random_int // self.precision) % 2))
        return (is_positive, is_human, seqname, coord + coord_start)


class FineTuneBiRen(SeqBERTLightningModule):

    def __init__(self, **hparams):
        model = SeqBERT(classify_only=True, n_class=1, **hparams)
        super().__init__(model, **hparams)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.hparams.sample_freq = int(self.hparams.seq_len * self.hparams.seq_len_sample_freq)
        self.hparams.min_len = int(self.hparams.seq_len * self.hparams.crop_factor)

        self.train_acc = pl.metrics.Accuracy(threshold=0)
        self.train_roc_auc = pl.metrics.functional.classification.auroc
        self.auc_fn = pl.metrics.functional.classification.auc
        self.val_acc = pl.metrics.Accuracy(threshold=0, compute_on_step=False)
        self.val_pr_curve = pl.metrics.PrecisionRecallCurve(compute_on_step=False, pos_label=1)
        self.val_roc_curve = pl.metrics.classification.ROC(compute_on_step=False, pos_label=1)
        self.test_results = BinaryPredictTensorMetric()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def seq_transform(self, seq):
        source = bioseq_to_index(seq)
        target = TOKENS_BP_IDX['n'] * torch.ones(self.hparams.seq_len, dtype=source.dtype)
        cropped = random_crop(source, self.hparams.min_len, self.hparams.seq_len)
        return random_seq_fill(cropped, target)

    def get_dataloader(self, is_sequential=False, include_intervals=None, randomize_freq=0.):
        label_randomizer = LabelRadomizer(randomize_freq)
        if include_intervals is not None:
            include_intervals = bed_from_file(include_intervals)
        train_data = StridedSequence(FastaFile(self.hparams.seq_file),
                    self.hparams.seq_len,
                    include_intervals=include_intervals,
                    seq_transform=self.seq_transform,
                    label_transform=label_randomizer.transform,
                    sequential=is_sequential,
                    sample_freq=self.hparams.sample_freq,
                    min_len=self.hparams.min_len)
        return train_data.get_data_loader(self.hparams.batch_size, self.hparams.num_workers)

    def train_dataloader(self):
        return self.get_dataloader(
                is_sequential=False,
                include_intervals=self.hparams.train_intervals,
                randomize_freq=self.hparams.train_randomize_prop)

    def val_dataloader(self):
        return self.get_dataloader(
                is_sequential=True,
                include_intervals=self.hparams.valid_intervals,
                randomize_freq=self.hparams.valid_randomize_prop)

    def test_dataloader(self):
        return self.get_dataloader(True, self.hparams.test_intervals)

    def training_step(self, batch, batch_idx):
        x, (is_positive, is_human, seqname, coord) = batch
        target = is_positive.float()
        predicted, latent, embedded = self.model.forward(x)
        loss = self.loss_fn(predicted, target)
        self.log('tr_acc', self.train_acc(predicted, target), prog_bar=True)
        self.log('tr_roc', self.train_roc_auc(predicted, target), prog_bar=True)
        self.log('tr_roc_old',roc_auc(torch.sigmoid(predicted.squeeze()), target), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, (is_positive, is_human, seqname, coord) = batch
        target = is_positive.to(torch.float)
        predicted, latent, embedded = self.model.forward(x)
        loss = self.loss_fn(predicted, target)
        self.val_pr_curve(predicted, target)
        self.val_roc_curve(predicted, target)
        return loss

    def validation_epoch_end(self, val_step_outputs):
        self.log('val_acc', self.val_acc.compute())
        precision, recall, _ = self.val_pr_curve.compute()
        self.log('val_pr', self.auc_fn(recall, precision), prog_bar=True)
        fpr, tpr, _ = self.val_roc_curve.compute()
        self.log('val_roc', self.auc_fn(fpr, tpr), prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, (is_positive, is_human, seqname, coord) = batch
        target = is_positive.to(torch.float)
        predicted, latent, embedded = self.model.forward(x)
        loss = self.loss_fn(predicted, target)
        self.test_results(predicted, target)
        try:
            print(self.train_roc_auc(predicted.flatten(), target.flatten()).item())
        except:
            pass
        return loss

    def test_epoch_end(self, val_step_outputs):
        scores = self.test_results.compute()
        print('Saving test scores to', self.hparams.test_out_file)
        torch.save(scores, self.hparams.test_out_file)

    @staticmethod
    def add_model_specific_args(parent_parser):
        super_parser = SeqBERTLightningModule.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser])
        """
        Define parameters that only apply to this model
        """
        #data params
        parser.add_argument('--seq_file', default='data/vista/all-enhancers.fa', type=str)
        parser.add_argument('--train_intervals', default='data/vista/human-enhancers-train.bed', type=str)
        parser.add_argument('--valid_intervals', default='data/vista/human-enhancers-valid.bed', type=str)
        parser.add_argument('--test_intervals', default='data/vista/human-enhancers-test.bed', type=str)
        parser.add_argument('--seq_len_source_multiplier', default=2., type=float)  # how much length to add when loading
        parser.add_argument('--crop_factor', default=0.2, type=float)  # gives min_len as a proportion of seq_len
        parser.add_argument('--seq_len_sample_freq', default=0.5, type=float)  # gives sample_freq in StridedSequence
        parser.add_argument('--train_randomize_prop', default=0., type=float)
        parser.add_argument('--valid_randomize_prop', default=0., type=float)
        return parser


if __name__ == '__main__':
    main(FineTuneBiRen)
