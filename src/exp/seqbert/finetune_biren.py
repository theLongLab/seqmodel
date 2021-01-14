import sys
sys.path.append('./src')
from argparse import ArgumentParser
import re
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything

from seqmodel.functional import bioseq_to_index, random_crop, random_seq_fill
from seqmodel.functional.log import roc_auc
from seqmodel.seqdata.iterseq import StridedSequence, FastaFile, bed_from_file
from exp.seqbert import TOKENS_BP_IDX
from exp.seqbert.model import SeqBERT, CheckpointEveryNSteps, PrintGradients
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


class FineTuneBiRen(LightningModule):

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = SeqBERT(**hparams)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.fasta = FastaFile(self.hparams.seq_file)
        self.hparams.sample_freq = int(self.hparams.seq_len * self.hparams.seq_len_sample_freq)
        self.hparams.min_len = int(self.hparams.seq_len * self.hparams.crop_factor)

    def load_pretrained_model(self, seqbert_obj):
        self.model.embedding = seqbert_obj.embedding
        self.model.transformer = seqbert_obj.transformer

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def data_transform(self, seq):
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
                    transform=self.data_transform,
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

    def print_progress(self, predicted, target, x, is_human, seqname, coord):
        print(seqname, coord, is_human, 'roc', roc_auc(torch.sigmoid(predicted.squeeze()), target))

    def training_step(self, batch, batch_idx):
        x, (is_positive, is_human, seqname, coord) = batch
        target = is_positive.to(torch.float)
        predicted, latent, embedded = self.model.forward(x)
        # remove dim 2 (seq) from predicted
        loss = self.loss_fn(predicted.squeeze(), target)
        if batch_idx % self.hparams.print_progress_freq == 0:
            self.print_progress(predicted, target, x, is_human, seqname, coord)
        return {'loss': loss, #'seqname': seqname[-1], 'coord': coord[-1],
                'log': {'train_loss': loss,} #'seqname': seqname[-1], 'coord': coord[-1],},
                }

    def validation_step(self, batch, batch_idx):
        x, (is_positive, is_human, seqname, coord) = batch
        target = is_positive.to(torch.float)
        predicted, latent, embedded = self.model.forward(x)
        # remove dim 2 (seq) from predicted
        loss = self.loss_fn(predicted.squeeze(), target)
        # if batch_idx % self.hparams.print_progress_freq == 0:
        print('Validation')
        self.print_progress(predicted, target, x, is_human, seqname, coord)
        return {'loss': loss,
                'log': {
                    'val_loss': loss,
                    # 'correct': acc_numbers,
                    # 'train_sample': str_train_sample,
                }}

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])
        # model params
        parser.add_argument('--n_class', default=1, type=int)
        parser.add_argument('--n_dims', default=256, type=int)
        parser.add_argument('--n_heads', default=1, type=int)
        parser.add_argument('--n_layers', default=1, type=int)
        parser.add_argument('--n_decode_layers', default=1, type=int)
        parser.add_argument('--feedforward_dims', default=512, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--position_embedding', default='Sinusoidal', type=str)

        # training params
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--num_workers', default=0, type=int)

        #data params
        parser.add_argument('--seq_file', default='data/vista/all-enhancers.fa', type=str)
        parser.add_argument('--train_intervals', default='data/vista/human-enhancers-train.bed', type=str)
        parser.add_argument('--valid_intervals', default='data/vista/human-enhancers-valid.bed', type=str)
        parser.add_argument('--test_intervals', default='data/vista/human-enhancers-test.bed', type=str)
        parser.add_argument('--seq_len', default=1000, type=int)
        parser.add_argument('--seq_len_source_multiplier', default=2., type=float)  # how much length to add when loading
        parser.add_argument('--crop_factor', default=0.2, type=float)  # gives min_len as a proportion of seq_len
        parser.add_argument('--seq_len_sample_freq', default=0.5, type=float)  # gives sample_freq in StridedSequence
        parser.add_argument('--print_progress_freq', default=1000, type=int)
        parser.add_argument('--save_checkpoint_freq', default=1000, type=int)
        parser.add_argument('--load_checkpoint_path', default=None, type=str)
        parser.add_argument('--load_pretrained_model', default=None, type=str)
        parser.add_argument('--train_randomize_prop', default=0., type=float)
        parser.add_argument('--valid_randomize_prop', default=0., type=float)
        return parser


def main():
    parent_parser = ArgumentParser(add_help=False)
    parser = FineTuneBiRen.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1)
    args = parser.parse_args()

    seed_everything(0)
    # defaults
    args.mode = 'classify'
    print(vars(args))
    if args.load_checkpoint_path is not None:
        model = FineTuneBiRen.load_from_checkpoint(args.load_checkpoint_path, **vars(args))
    elif args.load_pretrained_model is not None:
        model = FineTuneBiRen(**vars(args))
        pretrained = Pretrain.load_from_checkpoint(args.load_pretrained_model)
        model.load_pretrained_model(pretrained.model)
    else:
        model = FineTuneBiRen(**vars(args))
    # args.callbacks = [CheckpointEveryNSteps(args.save_checkpoint_freq), PrintGradients()]
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == '__main__':
    main()
