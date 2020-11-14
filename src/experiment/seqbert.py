import sys
sys.path.append('./src')
import os.path
from itertools import chain
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything

from seqmodel.model.conv import DilateConvEncoder, SeqFeedForward
from seqmodel.model.attention import SinusoidalPosition
from seqmodel.functional.mask import generate_mask, mask_randomize, mask_fill, mask_select
from seqmodel.seqdata.mapseq import RandomRepeatSequence
from seqmodel.seqdata.iterseq import StridedSequence, bed_from_file, FastaFile
from seqmodel.functional.transform import INDEX_TO_BASE, Compose, bioseq_to_index, \
                            single_split, permute
from seqmodel.functional.log import prediction_histograms, normalize_histogram, \
                            summarize, correct, accuracy_per_class, accuracy, \
                            summarize_weights_and_grads, tensor_stats_str


# from https://github.com/PyTorchLightning/pytorch-lightning/issues/2534
class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        total_batch_idx = trainer.total_batch_idx
        if total_batch_idx % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = "{}_{}_{}.ckpt".format(self.prefix, epoch, total_batch_idx)
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            print("Saving to", ckpt_path)
            trainer.save_checkpoint(ckpt_path)


class PrintGradients(pl.Callback):
    def __init__(self):
        print('zero grad callback loaded')

    def on_before_zero_grad(self, *args, **kwargs):
        print('zero grad callback')
        print(args, kwargs)


class SeqBERT(LightningModule):

    TOKENS_BP = INDEX_TO_BASE + [  # AGCT 0 1 2 3
        'n',  # 4 unknown base
        'm',  # 5 masked base
        '~',  # 6 classification token (always at start)
        'f',  # 7 output token at classification token position, indicates pretext task false
        't',  # 8 output token indicating pretext task is true
        ]
    MASK_TOKEN = 5
    CLS_TOKEN = 6
    CLS_OFFSET = 7
    # for mask
    NO_LOSS_INDEX = 0
    MASK_INDEX = 1
    RANDOM_INDEX = 2
    KEEP_INDEX = 3

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.tokens = self.TOKENS_BP
        embedding = nn.Embedding(len(self.tokens), self.hparams.n_dims)
        if self.hparams.position_embedding == 'Sinusoidal':
            self.embedding = nn.Sequential(
                embedding,
                SinusoidalPosition(self.hparams.n_dims, dropout=self.hparams.dropout,
                                    max_len=(self.hparams.seq_len + 1)),  # add 1 for cls token
                )
        else:
            self.embedding = embedding

        self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(self.hparams.n_dims, self.hparams.n_heads,
                                    self.hparams.feedforward_dims, self.hparams.dropout),
                self.hparams.n_layers)

        self.decoder = SeqFeedForward(self.hparams.n_dims, len(self.tokens),
                        hidden_layers=self.hparams.n_decode_layers - 1, activation_fn=nn.ReLU)

        self.loss_fn = nn.CrossEntropyLoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.mask_props = (self.hparams.mask_prop, self.hparams.random_prop, self.hparams.keep_prop)
        self.prev_loss = 10000.

    def configure_optimizers(self):
        return torch.optim.Adam(chain(self.embedding.parameters(), self.transformer.parameters(),
                                self.decoder.parameters()), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        if self.hparams.DEBUG_use_random_data:
            train_data = RandomRepeatSequence(self.hparams.seq_len, n_batch=10000,
                                n_repeats=self.hparams.DEBUG_random_n_repeats,
                                repeat_len=self.hparams.DEBUG_random_repeat_len)
            return torch.utils.data.DataLoader(train_data, batch_size=self.hparams.batch_size,
                                    shuffle=True, num_workers=self.hparams.num_workers)
        else:
            intervals = None
            if self.hparams.train_intervals is not None:
                intervals = bed_from_file(self.hparams.train_intervals)
            train_data = StridedSequence(FastaFile(self.hparams.seq_file),
                        self.hparams.seq_len, include_intervals=intervals,
                        transforms=bioseq_to_index, sequential=False)
        return train_data.get_data_loader(self.hparams.batch_size, self.hparams.num_workers)

    def val_dataloader(self):
        if self.hparams.DEBUG_use_random_data:
            valid_data = RandomRepeatSequence(self.hparams.seq_len, n_batch=100,
                                n_repeats=self.hparams.DEBUG_random_n_repeats,
                                repeat_len=self.hparams.DEBUG_random_repeat_len)
            return torch.utils.data.DataLoader(valid_data, batch_size=self.hparams.batch_size,
                                    shuffle=False, num_workers=self.hparams.num_workers)
        else:
            intervals = None
            if self.hparams.valid_intervals is not None:
                intervals = bed_from_file(self.hparams.valid_intervals)
            valid_data = StridedSequence(FastaFile(self.hparams.seq_file),
                        self.hparams.seq_len, include_intervals=intervals, 
                        transforms=bioseq_to_index, sequential=True)
        return valid_data.get_data_loader(self.hparams.batch_size, self.hparams.num_workers)

    def data_transform(self, x_batch):
        with torch.no_grad():
            # split into 2 halves, permute half of the halves
            first, last = single_split(x_batch)
            is_permuted, last = permute(last)
            # recombine with classification boolean at front, set to correct token indexes by adding offset
            cls_target = is_permuted.to(x_batch.dtype) + self.CLS_OFFSET
            return torch.cat((cls_target.unsqueeze(1), first, last), dim=1)

    def mask_transform(self, target):
        with torch.no_grad():
            # randomize, mask, or mark for loss calculation some proportion of positions
            mask = generate_mask(target, self.mask_props)
            # omit classification token (1st position in seq) from masked prediction loss
            mask[:, 0] = self.NO_LOSS_INDEX
            source = mask_randomize(target, mask == self.RANDOM_INDEX, 4)  # number of random tokens = base pairs
            source = mask_fill(source, mask == self.MASK_INDEX, self.MASK_TOKEN)
            # make the first position the classification token
            source[:, 0] = self.CLS_TOKEN
            # return mask of all positions that will contribute to loss
            return source, target, mask

    def masked_forward(self, x):
        target = self.data_transform(x)
        source, target, mask = self.mask_transform(target)
        # input dims are (batch, seq), embedding adds channel dim to end
        # swap dimensions from (batch, seq, channel) to (seq, batch, channel)
        embedded = self.embedding(source).permute(1, 0, 2)
        latent = self.transformer(embedded)
        # swap dimensions from (seq, batch, channel) to (batch, channels, seq_len)
        predicted = self.decoder(latent.permute(1, 2, 0))
        masked_predict_loss = self.loss_fn(mask_select(predicted, mask != self.NO_LOSS_INDEX),
                                            mask_select(target, mask != self.NO_LOSS_INDEX))
        # apply classification loss separately
        cls_loss = self.loss_fn(predicted[:,:, 0], target[:, 0])
        loss = masked_predict_loss + self.hparams.cls_regularization * cls_loss
        return loss, masked_predict_loss, cls_loss, predicted, latent, source, target, mask, embedded

    def forward(self, batch):
        x, (seqname, coord) = batch
        # swap dimensions from (batch, seq, channel) to (seq, batch, channel)
        x_1 = self.embedding(x).permute(1, 0, 2)
        z = self.transformer(x_1)
        # swap dimensions from (seq, batch, channel) to (batch, channels, seq_len)
        y = self.decoder(z.permute(1, 2, 0))
        return y

    def training_step(self, batch, batch_idx):
        x, (seqname, coord) = batch
        loss, pred_loss, cls_loss, predicted, latent, source, target, mask, embedded = self.masked_forward(x)
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

    def print_progress(self, loss, predicted, latent, source, target, mask, embedded, seqname, coord, include_grad=False):
        str_train_sample = summarize(
            mask + len(self.tokens),
            source,
            target,
            correct(predicted, target),
            predicted.permute(1, 0, 2),
            index_symbols=self.tokens + [' ', '_', '?', '='])

        # if include_grad:
        #     loss.backward(retain_graph=True)
        #     self.optimizers().step()
        # print(summarize_weights_and_grads({'embedding': self.embedding, 'transformer': self.transformer,
        #         'decoder': self.decoder}, include_grad=False, threshold_trigger=0.))
        hist = prediction_histograms(predicted.detach().cpu(), target.detach().cpu(), n_bins=5)
        acc = normalize_histogram(hist)
        acc_numbers = accuracy_per_class(hist, threshold_prob=1. / len(self.tokens))
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
        parser.add_argument('--cls_regularization', default=1., type=float)
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
        parser.add_argument('--save_checkpoint_freq', default=1000, type=int)
        parser.add_argument('--load_checkpoint_path', default=None, type=str)
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
    if args.load_checkpoint_path is not None:
        model = SeqBERT.load_from_checkpoint(args.load_checkpoint_path, seq_file=args.seq_file,
                    train_intervals=args.train_intervals, valid_intervals=args.valid_intervals)
    args.callbacks = [CheckpointEveryNSteps(args.save_checkpoint_freq), PrintGradients()]
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == '__main__':
    main()
