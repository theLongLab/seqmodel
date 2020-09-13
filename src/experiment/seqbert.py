import sys
sys.path.append('./src')
import os.path
from itertools import chain
from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything

from seqmodel.model.conv import DilateConvEncoder, SeqFeedForward
from seqmodel.model.attention import SinusoidalPosition
from seqmodel.functional.mask import PositionMask
from seqmodel.seqdata.mapseq import RandomRepeatSequence
from seqmodel.seqdata.iterseq import StridedSequence, bed_from_file,fasta_from_file
from seqmodel.functional.transform import INDEX_TO_BASE, Compose, one_hot_to_index
from seqmodel.task.log import prediction_histograms, normalize_histogram, \
                            summarize, correct, accuracy_per_class


class SeqBERT(LightningModule):

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.position_embedding == 'Sinusoidal':
            self.embedding = nn.Sequential(
                nn.Embedding(4, self.hparams.n_dims),
                SinusoidalPosition(self.hparams.n_dims, dropout=self.hparams.dropout,
                                    max_len=self.hparams.seq_len),
                )
        else:
            self.embedding = nn.Embedding(4, self.hparams.n_dims)
        self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(self.hparams.n_dims, self.hparams.n_heads,
                                    self.hparams.feedforward_dims, self.hparams.dropout),
                self.hparams.n_layers)
        self.decoder = SeqFeedForward(self.hparams.n_dims, 4, hidden_layers=self.hparams.n_decode_layers - 1,
                                activation_fn=nn.ReLU)
        self.loss_fn = nn.CrossEntropyLoss()
        self.mask = PositionMask(mask_prop=self.hparams.mask_prop, random_prop=self.hparams.random_prop,
                                    keep_prop=self.hparams.keep_prop,)

    def configure_optimizers(self):
        return torch.optim.Adam(chain(self.embedding.parameters(), self.transformer.parameters(),
                                self.decoder.parameters()), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        if self.hparams.DEBUG_use_random_data:
            train_data = RandomRepeatSequence(self.hparams.seq_len, n_batch=10000,
                                n_repeats=self.hparams.DEBUG_random_n_repeats,
                                repeat_len=self.hparams.DEBUG_random_repeat_len)
        else:
            # train_data = MapSequence.from_file('data/ref_genome/chr22_excerpt_4m.fa', 500, remove_gaps=True)
            intervals = None
            if self.hparams.train_intervals is not None:
                intervals = bed_from_file(self.hparams.train_intervals)
            train_data = StridedSequence(
                self.hparams.seq_file, self.hparams.seq_len, include_intervals=intervals, sequential=False)
        return train_data.get_data_loader(self.hparams.batch_size, self.hparams.num_workers)

    def val_dataloader(self):
        if self.hparams.DEBUG_use_random_data:
            valid_data = RandomRepeatSequence(self.hparams.seq_len, n_batch=100,
                                n_repeats=self.hparams.DEBUG_random_n_repeats,
                                repeat_len=self.hparams.DEBUG_random_repeat_len)
        else:
            # valid_data = MapSequence.from_file('data/ref_genome/chr22_excerpt_800k.fa', 500, remove_gaps=True)
            intervals = None
            if self.hparams.valid_intervals is not None:
                intervals = bed_from_file(self.hparams.valid_intervals)
            valid_data = StridedSequence(
                self.hparams.seq_file, self.hparams.seq_len, include_intervals=intervals, sequential=True)
        return valid_data.get_data_loader(self.hparams.batch_size, self.hparams.num_workers)

    def forward(self, batch):
        x, seqname, coord = batch
        # swap dimensions from (batch, seq, channel) to (seq, batch, channel)
        x_1 = self.embedding(x).permute(1, 0, 2)
        z = self.transformer(x_1)
        # swap dimensions from (seq, batch, channel) to (batch, channels, seq_len)
        y = self.decoder(z.permute(1, 2, 0))
        return y

    def masked_forward(self, x_in):
        # swap dimensions from (batch, seq, channel) to (seq, batch, channel)
        x, mask = self.mask.attn_mask(x_in, mask_value=True, mask_fill=True)
        latent = self.transformer(self.embedding(x).permute(1, 0, 2), src_key_padding_mask=mask)
        # swap dimensions from (seq, batch, channel) to (batch, channels, seq_len)
        predicted = self.decoder(latent.permute(1, 2, 0))
        loss = self.loss_fn(*self.mask.select(predicted, x_in))
        return loss, predicted, latent, x

    def training_step(self, batch, batch_idx):
        x_in, seqname, coord = batch
        loss, predicted, _, x = self.masked_forward(x_in)
        if batch_idx % self.hparams.print_progress_freq == 0:
            self.print_progress(predicted, x_in, x)
        return {'loss': loss, #'seqname': seqname[-1], 'coord': coord[-1],
                'log': {'train_loss': loss,} #'seqname': seqname[-1], 'coord': coord[-1],},
                }

    def print_progress(self, predicted, x_in, x):
        str_train_sample = summarize(self.mask.mask_val + 4, x, correct(predicted, x_in),
                predicted.permute(1, 0, 2), index_symbols=INDEX_TO_BASE + [' ', '_', '?', '='])
        hist = prediction_histograms(predicted.detach().cpu(), x_in.detach().cpu(), n_bins=5)
        acc = normalize_histogram(hist)
        acc_numbers = accuracy_per_class(hist, threshold_prob=0.5)
        str_acc = summarize(acc, col_labels=INDEX_TO_BASE, normalize_fn=None)
        print(acc_numbers, str_acc, str_train_sample, sep='\n')

    def validation_step(self, batch, batch_idx):
        x_in, seqname, coord = batch
        loss, predicted, _, x = self.masked_forward(x_in)
        if batch_idx % self.hparams.print_progress_freq == 0:
            self.print_progress(predicted, x_in, x)
        return {'loss': loss,
                'log': {
                    'val_loss': loss,
                    # 'correct': acc_numbers,
                    # 'train_sample': str_train_sample,
                }}

    # def validation_epoch_end(self, val_step_outputs):
    #     result = pl.EvalResult(checkpoint_on=loss)
    #     result.log('val_loss', loss)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])
        # model params
        parser.add_argument('--n_dims', default=256, type=int)
        parser.add_argument('--n_heads', default=1, type=int)
        parser.add_argument('--n_layers', default=1, type=int)
        parser.add_argument('--n_decode_layers', default=1, type=int)
        parser.add_argument('--feedforward_dims', default=512, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--position_embedding', default='Sinusoidal', type=str)

        # training params
        parser.add_argument('--keep_prop', default=0.05, type=float)
        parser.add_argument('--mask_prop', default=0.08, type=float)
        parser.add_argument('--random_prop', default=0.02, type=float)
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)

        #data params
        parser.add_argument('--seq_file', default='data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa', type=str)
        parser.add_argument('--train_intervals', default=None, type=str)
        parser.add_argument('--valid_intervals', default=None, type=str)
        parser.add_argument('--seq_len', default=500, type=int)
        parser.add_argument('--print_progress_freq', default=1000, type=int)
        parser.add_argument('--DEBUG_use_random_data', default=False, type=bool)
        parser.add_argument('--DEBUG_random_repeat_len', default=1, type=int)
        parser.add_argument('--DEBUG_random_n_repeats', default=500, type=int)
        return parser


def main():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    parser = SeqBERT.add_model_specific_args(parent_parser, root_dir)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1)
    args = parser.parse_args()

    seed_everything(0)
    print(args)
    model = SeqBERT(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == '__main__':
    main()
