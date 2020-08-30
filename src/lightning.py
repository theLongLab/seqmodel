import os.path
from itertools import chain
from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything

from seqmodel.model.conv import DilateConvEncoder, SeqFeedForward
from seqmodel.task.task import LambdaLoss
from seqmodel.task.mask import PositionMask
from seqmodel.seq.mapseq import MapSequence, RandomRepeatSequence
from seqmodel.seq.transform import INDEX_TO_BASE, Compose, one_hot_to_index
from seqmodel.task.log import prediction_histograms, normalize_histogram, \
                            summarize, correct, accuracy_per_class

def print_and_pass(x):
    print(x.shape, torch.min(x), torch.std_mean(x), torch.max(x))
    return x

class SeqBERT(LightningModule):

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        # encoder = DilateConvEncoder(4, 3, 2, 2., 1, 3, 0.1)
        # decoder = SeqFeedForward(encoder.out_channels, 4, hidden_layers=self.hparams.n_decode_layers - 1,
        #                         activation_fn=nn.ReLU)
        self.embedding = nn.Embedding(4, self.hparams.n_dims)
        self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(self.hparams.n_dims, self.hparams.n_heads,
                                    self.hparams.feedforward_dims, self.hparams.dropout),
                self.hparams.n_layers)
        self.decoder = SeqFeedForward(self.hparams.n_dims, 4, hidden_layers=self.hparams.n_decode_layers - 1,
                                activation_fn=nn.ReLU)
        self.loss_fn = nn.CrossEntropyLoss()
        self.mask = PositionMask(keep_prop=self.hparams.keep_prop,
                            mask_prop=self.hparams.mask_prop,
                            random_prop=self.hparams.random_prop)

    def configure_optimizers(self):
        return torch.optim.Adam(chain(self.embedding.parameters(), self.transformer.parameters(),
                                self.decoder.parameters()), lr=self.hparams.learning_rate)

    def forward(self, x):
        # swap dimensions from (batch, seq, channel) to (seq, batch, channel)
        x_1 = self.embedding(x).permute(1, 0, 2)
        z = self.transformer(x_1)
        # swap dimensions from (seq, batch, channel) to (batch, channels, seq_len)
        y = self.decoder(z.permute(1, 2, 0))
        return y

    def masked_forward(self, batch):
        # swap dimensions from (batch, seq, channel) to (seq, batch, channel)
        x, mask = self.mask.attn_mask(batch, mask_value=True, mask_fill=False)
        latent = self.transformer(self.embedding(x).permute(1, 0, 2), src_key_padding_mask=mask)
        # swap dimensions from (seq, batch, channel) to (batch, channels, seq_len)
        predicted = self.decoder(latent.permute(1, 2, 0))
        loss = self.loss_fn(*self.mask.select(predicted, batch))
        return loss, predicted, latent, x

    def training_step(self, batch, batch_idx):
        loss, predicted, _, _ = self.masked_forward(batch)
        return {'loss': loss,
                'log': {'train_loss': loss}}

    def train_dataloader(self):
        if self.hparams.DEBUG_use_random_data:
            train_data = RandomRepeatSequence(500, n_batch=10000, n_repeats=500,
                                repeat_len=self.hparams.DEBUG_random_repeat_len)
        else:
            train_data = MapSequence.from_file('data/ref_genome/chr22_excerpt_4m.fa', 500, remove_gaps=True)
        return torch.utils.data.DataLoader(train_data, batch_size=self.hparams.batch_size,
                            shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        if self.hparams.DEBUG_use_random_data:
            valid_data = RandomRepeatSequence(500, n_batch=100, n_repeats=500,
                                repeat_len=self.hparams.DEBUG_random_repeat_len)
        else:
            valid_data = MapSequence.from_file('data/ref_genome/chr22_excerpt_800k.fa', 500, remove_gaps=True)
        return torch.utils.data.DataLoader(valid_data, batch_size=self.hparams.batch_size,
                            shuffle=False, num_workers=self.hparams.num_workers)

    def validation_step(self, batch, batch_idx):
        loss, predicted, _, _ = self.masked_forward(batch)
        str_train_sample = summarize(self.mask.mask_val + 4, batch, correct(predicted, batch),
                predicted.permute(1, 0, 2), index_symbols=INDEX_TO_BASE + [' ', '_', '?', '='])
        # hist = prediction_histograms(predicted, batch, n_bins=3)
        # acc = normalize_histogram(hist)
        # acc_numbers = accuracy_per_class(hist)
        # str_acc = summarize(acc, col_labels=INDEX_TO_BASE, normalize_fn=None)
        # print(acc_numbers, str_acc, str_train_sample, sep='\n')
        print('', str_train_sample, sep='\n')
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

        # task params
        parser.add_argument('--keep_prop', default=0.05, type=float)
        parser.add_argument('--mask_prop', default=0.08, type=float)
        parser.add_argument('--random_prop', default=0.02, type=float)

        # training params
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)

        #data params
        parser.add_argument('--DEBUG_use_random_data', default=False, type=bool)
        parser.add_argument('--DEBUG_random_repeat_len', default=1, type=int)
        return parser


class LambdaUnsupervisedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], None


def main(args):
    seed_everything(0)
    model = SeqBERT(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


def run_cli():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    parser = SeqBERT.add_model_specific_args(parent_parser, root_dir)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1)
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
