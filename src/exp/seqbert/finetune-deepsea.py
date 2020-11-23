import sys
sys.path.append('./src')
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything

from seqmodel.functional.transform import one_hot_to_index
from seqmodel.functional.log import roc_auc
from selene.mat_file_sampler import MatFileSampler
from exp.seqbert.model import SeqBERT, CheckpointEveryNSteps, PrintGradients
from exp.seqbert.pretrain import Pretrain


class MatFileDataset(IterableDataset):

    def __init__(self, sampler, batch_size, cls_token):
        self.sampler = sampler
        self.batch_size = batch_size
        self.cls_token = cls_token

    def __iter__(self):
        for _ in range(self.sampler.n_samples):
            batch, _ = self.sampler.get_data_and_targets(self.batch_size, n_samples=self.batch_size)
            seq, target = batch[0]
            # swap dimensions from (batch, seq, channel) to the usual (batch, channel, seq)
            seq = torch.tensor(seq, dtype=torch.long).permute(0, 2, 1)
            seq = one_hot_to_index(seq)  # embedding works on indices
            target = torch.tensor(target, dtype=torch.float)
            cls_tokens = torch.zeros([seq.shape[0], 1], dtype=torch.long) + self.cls_token
            seq = torch.cat([cls_tokens, seq], dim=1)
            yield seq, target  # (batch, seq, channel) and (batch, channel)


class FineTuneDeepSEA(LightningModule):

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = SeqBERT(**hparams)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def load_pretrained_model(self, seqbert_obj):
        self.model.embedding = seqbert_obj.embedding
        self.model.transformer = seqbert_obj.transformer

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        train_data = MatFileSampler(self.hparams.train_mat, 'trainxdata', 'traindata',
            sequence_batch_axis=2, sequence_alphabet_axis=1, targets_batch_axis=1)
        return MatFileDataset(train_data, self.hparams.batch_size, self.model.CLS_TOKEN)

    def val_dataloader(self):
        valid_data = MatFileSampler(self.hparams.valid_mat, 'validxdata', 'validdata',
            sequence_batch_axis=0, sequence_alphabet_axis=1, targets_batch_axis=0, shuffle=False)
        return MatFileDataset(valid_data, self.hparams.batch_size, self.model.CLS_TOKEN)

    def test_dataloader(self):
        test_data = MatFileSampler(self.hparams.test_mat, 'testxdata', 'testdata',
            sequence_batch_axis=0, sequence_alphabet_axis=1, targets_batch_axis=0, shuffle=False)
        return MatFileDataset(test_data, self.hparams.batch_size, self.model.CLS_TOKEN)

    def training_step(self, batch, batch_idx):
        x, target = batch
        predicted, latent, embedded = self.model.forward(x)
        loss = self.loss_fn(predicted.squeeze(), target)
        if batch_idx % self.hparams.print_progress_freq == 0:
            self.print_progress(predicted, target, x)
        return {'loss': loss, #'seqname': seqname[-1], 'coord': coord[-1],
                'log': {'train_loss': loss,} #'seqname': seqname[-1], 'coord': coord[-1],},
                }

    def print_progress(self, predicted, target, x):
        print('roc', roc_auc(torch.sigmoid(predicted.squeeze()), target))

    def validation_step(self, batch, batch_idx):
        x, target = batch
        predicted, latent, embedded = self.model.forward(x)
        # remove dim 2 (seq) from predicted
        loss = self.loss_fn(predicted.squeeze(), target)
        if batch_idx % self.hparams.print_progress_freq == 0:
            self.print_progress(predicted, target, x)
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
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])
        # model params
        parser.add_argument('--n_class', default=919, type=int)
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

        #data params
        parser.add_argument('--train_mat', default='data/deepsea/train.mat', type=str)
        parser.add_argument('--valid_mat', default='data/deepsea/valid.mat', type=str)
        parser.add_argument('--test_mat', default='data/deepsea/test.mat', type=str)
        parser.add_argument('--seq_len', default=1000, type=int)
        parser.add_argument('--print_progress_freq', default=1000, type=int)
        parser.add_argument('--save_checkpoint_freq', default=1000, type=int)
        parser.add_argument('--load_checkpoint_path', default=None, type=str)
        parser.add_argument('--load_pretrained_model', default=None, type=str)
        return parser


def main():
    parent_parser = ArgumentParser(add_help=False)
    parser = FineTuneDeepSEA.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1)
    args = parser.parse_args()

    seed_everything(0)
    # defaults
    args.mode = 'classify'
    print(vars(args))
    if args.load_checkpoint_path is not None:
        model = FineTuneDeepSEA.load_from_checkpoint(args.load_checkpoint_path, **vars(args))
    elif args.load_pretrained_model is not None:
        model = FineTuneDeepSEA(**vars(args))
        pretrained = Pretrain.load_from_checkpoint(args.load_pretrained_model)
        model.load_pretrained_model(pretrained.model)
    else:
        model = FineTuneDeepSEA(**vars(args))
    args.callbacks = [CheckpointEveryNSteps(args.save_checkpoint_freq), PrintGradients()]
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == '__main__':
    main()
