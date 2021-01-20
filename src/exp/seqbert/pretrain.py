import sys
sys.path.append('./src')
import os.path
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything

from seqmodel import INDEX_TO_BASE
from seqmodel.functional.mask import generate_mask, mask_randomize, mask_fill, mask_select
from seqmodel.seqdata.mapseq import RandomRepeatSequence
from seqmodel.seqdata.iterseq import StridedSequence, bed_from_file, FastaFile
from seqmodel.functional import Compose, bioseq_to_index, permute
from seqmodel.functional.log import prediction_histograms, normalize_histogram, \
                            summarize, correct, accuracy_per_class, accuracy, \
                            summarize_weights_and_grads, tensor_stats_str
from exp.seqbert import TOKENS_BP_IDX
from exp.seqbert.model import SeqBERT, CheckpointEveryNSteps, bool_to_tokens


class PretrainBatchProcessor():

    def __init__(self, seq_len,
                min_len, max_len,
                offset_min, offset_max,
                mask_prop, random_prop, keep_prop):
        self.seq_len = seq_len
        self.min_len = min_len
        self.max_len = max_len
        self.offset_min = offset_min
        self.offset_max = offset_max
        self.mask_props = (mask_prop, random_prop, keep_prop)

    def split_shuffle(self, batch):
        with torch.no_grad():
            # split into first and last subsequences, permute half of 'last' subsequences
            first, last = torch.split(batch, (batch.size(1) // 2,
                    batch.size(1) - (batch.size(1) // 2)), dim=1)
            is_permuted, last = permute(last, prop=0.5)
            # labels: True (not permuted) or False (permuted) token
            cls_target = bool_to_tokens(torch.logical_not(is_permuted))
            # insert sep token between 'first' and 'last'
            sep = torch.ones([batch.size(0), 1], dtype=batch.dtype) * TOKENS_BP_IDX['/']
            return cls_target, torch.cat([first, sep, last], dim=1)

    def rand_subseq(self, split_seqs):  # index relative to source midpoint
        batch_size = split_seqs.size(0)
        src_midpoint = split_seqs.size(1) // 2  # assume SEP token at midpoint
        # crop positions of first and last half
        starts = -1 * torch.randint(self.min_len, self.max_len, [batch_size])
        ends = torch.randint(self.min_len, self.max_len, [batch_size]) + 1 # add 1 for SEP token
        # position of SEP token in tgt
        sep_offsets = torch.randint(self.offset_min, self.offset_max, [batch_size])

        # fill target with empty/undefined base 'N'
        target = TOKENS_BP_IDX['n'] * torch.ones([batch_size, self.seq_len],
                                                        dtype=split_seqs.dtype)
        for i, (seq, start, end, offset) in enumerate(zip(split_seqs, starts, ends, sep_offsets)):
            tgt_start = max(1, start + offset)  # position 0 reserved for CLS token
            tgt_end = min(self.seq_len, end + offset)
            src_start = tgt_start - offset + src_midpoint
            src_end = tgt_end - offset + src_midpoint
            target[i, tgt_start:tgt_end] = seq[src_start:src_end]
        return target, sep_offsets

    def mask_transform(self, target):
        with torch.no_grad():
            # randomize, mask, or mark for loss calculation some proportion of positions
            mask = generate_mask(target, self.mask_props)
            # omit classification token, separator, and any empty 'N'
            mask[:, 0] = Pretrain.NO_LOSS_INDEX
            omit = torch.logical_or(target == TOKENS_BP_IDX['/'],
                                    target == TOKENS_BP_IDX['n'])
            mask = mask.masked_fill(omit, Pretrain.NO_LOSS_INDEX)
            source = mask_randomize(target, mask == Pretrain.RANDOM_INDEX, 4)  # 4 base pairs
            source = mask_fill(source, mask == Pretrain.MASK_INDEX, TOKENS_BP_IDX['m'])
            # replace the classification target with CLS token '~'
            source[:, 0] = TOKENS_BP_IDX['~']
            # return mask of all positions that will contribute to loss
            return source, mask

    def collate(self, samples):
        sequences, metadata = zip(*samples)  # make each datatype a separate list
        key, coord = zip(*metadata)
        # shuffle for next sequence prediction task
        cls_targets, split_seqs = self.split_shuffle(torch.stack(sequences, dim=0))
        target, _ = self.rand_subseq(split_seqs)
        target[:, 0] = cls_targets
        # mask for masked token prediction task
        source, mask = self.mask_transform(target)
        return (source, target, mask), (key, torch.tensor(coord))  # send this to GPU


class Pretrain(LightningModule):

    # for mask
    NO_LOSS_INDEX = 0
    MASK_INDEX = 1
    RANDOM_INDEX = 2
    KEEP_INDEX = 3

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = SeqBERT(**hparams)
        self.loss_fn = nn.CrossEntropyLoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.prev_loss = 10000.

        # get sequence of length 2*seq_len from dataloader
        self.load_seq_len = self.hparams.seq_len_source_multiplier * self.hparams.seq_len
        min_crop = int(self.hparams.seq_len * self.hparams.crop_factor)
        max_crop = self.hparams.seq_len - min_crop
        offset_min = 1 + min_crop
        offset_max = self.hparams.seq_len - min_crop
        self.batch_processor = PretrainBatchProcessor(self.hparams.seq_len,
                min_crop, max_crop, offset_min, offset_max,
                self.hparams.mask_prop, self.hparams.random_prop, self.hparams.keep_prop)
        self.sample_freq = int(self.hparams.seq_len * self.hparams.seq_len_sample_freq)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        if self.hparams.DEBUG_use_random_data:
            train_data = RandomRepeatSequence(self.load_seq_len, n_batch=10000,
                                n_repeats=self.hparams.DEBUG_random_n_repeats,
                                repeat_len=self.hparams.DEBUG_random_repeat_len)
            return torch.utils.data.DataLoader(train_data, batch_size=self.hparams.batch_size,
                                shuffle=True, num_workers=self.hparams.num_workers,
                                collate_fn=self.batch_processor.collate)
        else:
            intervals = None
            if self.hparams.train_intervals is not None:
                intervals = bed_from_file(self.hparams.train_intervals)
            train_data = StridedSequence(FastaFile(self.hparams.seq_file),
                        self.load_seq_len, include_intervals=intervals,
                        transform=bioseq_to_index, sequential=False,
                        sample_freq=self.sample_freq)
        return train_data.get_data_loader(self.hparams.batch_size, self.hparams.num_workers,
                        collate_fn=self.batch_processor.collate)

    def val_dataloader(self):
        if self.hparams.DEBUG_use_random_data:
            valid_data = RandomRepeatSequence(self.load_seq_len, n_batch=100,
                                n_repeats=self.hparams.DEBUG_random_n_repeats,
                                repeat_len=self.hparams.DEBUG_random_repeat_len)
            return torch.utils.data.DataLoader(valid_data, batch_size=self.hparams.batch_size,
                                shuffle=False, num_workers=self.hparams.num_workers,
                                collate_fn=self.batch_processor.collate)
        else:
            intervals = None
            if self.hparams.valid_intervals is not None:
                intervals = bed_from_file(self.hparams.valid_intervals)
            valid_data = StridedSequence(FastaFile(self.hparams.seq_file),
                        self.load_seq_len, include_intervals=intervals, 
                        transform=bioseq_to_index, sequential=True,
                        sample_freq=self.sample_freq)
        return valid_data.get_data_loader(self.hparams.batch_size, self.hparams.num_workers,
                        collate_fn=self.batch_processor.collate)

    def masked_forward(self, batch):
        source, target, mask = batch
        predicted, latent, embedded = self.model.forward(source)
        masked_predict_loss = self.loss_fn(mask_select(predicted, mask != self.NO_LOSS_INDEX),
                                            mask_select(target, mask != self.NO_LOSS_INDEX))
        # apply classification loss separately
        cls_loss = self.loss_fn(predicted[:,:, 0], target[:, 0])
        loss = masked_predict_loss + self.hparams.cls_regularization * cls_loss
        return loss, masked_predict_loss, cls_loss, predicted, latent, source, target, mask, embedded

    def training_step(self, x, batch_idx):
        batch, (seqname, coord) = x
        loss, pred_loss, cls_loss, predicted, latent, source, target, mask, embedded = self.masked_forward(batch)
        print('pred: %2.4f cls: %2.4f' % (pred_loss.item(), cls_loss.item()))
        # if batch_idx > 1000 and loss > 2.2:
        #     print(loss, self.prev_loss)
        #     self.print_progress(loss, predicted, latent, source, target, mask, embedded, seqname, coord, include_grad=True)
        #     raise Exception("loss delta exceeded")
        self.prev_loss = loss.item()
        if batch_idx % self.hparams.print_progress_freq == 0:
            self.print_progress(loss, predicted, latent, source, target, mask, embedded, seqname, coord, include_grad=True)
        return {'loss': loss, #'seqname': seqname[-1], 'coord': coord[-1],
                'log': {'train_loss': loss,} #'seqname': seqname[-1], 'coord': coord[-1],},
                }

    def accuracy_by_type():
        pass # TODO
        # mask_select(predicted, mask == self.MASK_INDEX)
        # mask_select(predicted, mask == self.RANDOM_INDEX)
        # mask_select(predicted, mask == self.KEEP_INDEX)
        # mask_select(predicted, mask == self.NO_LOSS_INDEX)

    def print_progress(self, loss, predicted, latent, source, target, mask, embedded, seqname, coord, include_grad=False):
        str_train_sample = summarize(
            mask + len(self.model.tokens),
            source,
            target,
            correct(predicted, target),
            predicted.permute(1, 0, 2),
            index_symbols=self.model.tokens + [' ', '_', '?', '='])  # extra symbols represent masking

        # if include_grad:
        #     loss.backward(retain_graph=True)
        #     self.optimizers().step()
        # print(summarize_weights_and_grads({'embedding': self.embedding, 'transformer': self.transformer,
        #         'decoder': self.decoder}, include_grad=False, threshold_trigger=0.))
        hist = prediction_histograms(predicted.detach().cpu(), target.detach().cpu(), n_bins=5)
        acc = normalize_histogram(hist)
        acc_numbers = accuracy_per_class(hist, threshold_prob=1. / len(self.model.tokens))
        str_acc = summarize(acc, col_labels=INDEX_TO_BASE, normalize_fn=None)
        cls_acc = accuracy(predicted[:,:, 0], target[:, 0])
        print(seqname[0], coord[0], cls_acc, acc_numbers)
        print(str_acc, str_train_sample, sep='\n')

        embedded_vector_lengths = torch.norm(embedded, dim=2)  # vector length along channel dim
        latent_vector_lengths = torch.norm(latent, dim=2)  # vector length along channel dim
        embed_reshape = embedded.reshape(-1, embedded.shape[2])
        latent_reshape = latent.reshape(-1, latent.shape[2])
        embedded_pairwise_dist = F.pairwise_distance(embed_reshape[:, :-1], embed_reshape[:, 1:])
        latent_pairwise_dist = F.pairwise_distance(latent_reshape[:, :-1], latent_reshape[:, 1:])
        print('embedding/latent', tensor_stats_str(
            embedded_vector_lengths, embedded_pairwise_dist, latent_vector_lengths, latent_pairwise_dist))

    def validation_step(self, batch, batch_idx):
        x, (seqname, coord) = batch
        loss, pred_loss, cls_loss, predicted, latent, source, target, mask, embedded = self.masked_forward(x)
        if batch_idx % self.hparams.print_progress_freq == 0:
            self.print_progress(loss, predicted, latent, source, target, mask, embedded, seqname, coord)
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
        parser.add_argument('--n_dims', default=256, type=int)
        parser.add_argument('--n_heads', default=1, type=int)
        parser.add_argument('--n_layers', default=1, type=int)
        parser.add_argument('--n_decode_layers', default=1, type=int)
        parser.add_argument('--feedforward_dims', default=512, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--cls_regularization', default=1., type=float)
        parser.add_argument('--position_embedding', default='Sinusoidal', type=str)

        # training params
        parser.add_argument('--keep_prop', default=0.05, type=float)
        parser.add_argument('--mask_prop', default=0.08, type=float)
        parser.add_argument('--random_prop', default=0.02, type=float)
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)

        # data params
        parser.add_argument('--seq_file', default='data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa', type=str)
        parser.add_argument('--train_intervals', default=None, type=str)
        parser.add_argument('--valid_intervals', default=None, type=str)
        parser.add_argument('--seq_len', default=500, type=int)
        parser.add_argument('--seq_len_source_multiplier', default=2., type=float)  # how much length to add when loading
        parser.add_argument('--crop_factor', default=0.2, type=float)  # how much of source sequence to keep when cropping
        parser.add_argument('--seq_len_sample_freq', default=0.5, type=float)  # gives sample_freq in StridedSequence

        parser.add_argument('--print_progress_freq', default=1000, type=int)
        parser.add_argument('--save_checkpoint_freq', default=1000, type=int)
        parser.add_argument('--load_checkpoint_path', default=None, type=str)
        parser.add_argument('--DEBUG_use_random_data', default=False, type=bool)
        parser.add_argument('--DEBUG_random_repeat_len', default=1, type=int)
        parser.add_argument('--DEBUG_random_n_repeats', default=500, type=int)
        return parser


def main():
    parent_parser = ArgumentParser(add_help=False)

    parser = Pretrain.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1)
    args = parser.parse_args()

    seed_everything(0)
    # defaults
    args.mode = 'all'
    args.n_class = -1
    print(vars(args))
    if args.load_checkpoint_path is not None:
        model = Pretrain.load_from_checkpoint(args.load_checkpoint_path, **vars(args))
    else:
        model = Pretrain(**vars(args))
    args.callbacks = [CheckpointEveryNSteps(args.save_checkpoint_freq)]
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == '__main__':
    main()
