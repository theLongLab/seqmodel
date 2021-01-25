import sys
sys.path.append('./src')
import os.path
from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import GPUStatsMonitor


from seqmodel.model.conv import SeqFeedForward
from seqmodel.model.attention import SinusoidalPosition
from exp.seqbert import TOKENS_BP


def bool_to_tokens(bool_tensor, target_tensor_type=torch.long):
    return bool_tensor.to(target_tensor_type) + 7  # 'f' token is 0 + 7, so 1 + 7 = 8 is true


class SeqBERT(nn.Module):

    def __init__(self, classify_only=False, n_class=None, **hparams):
        super().__init__()
        self.tokens = TOKENS_BP  # may have different tokenizations in the future
        embedding = nn.Embedding(len(self.tokens), hparams['n_dims'])
        if hparams['position_embedding'] == 'Sinusoidal':
            self.embedding = nn.Sequential(
                embedding,
                SinusoidalPosition(hparams['n_dims'], dropout=hparams['dropout'],
                                    max_len=(hparams['seq_len'] + 1)),  # add 1 for cls token
                )
        else:
            self.embedding = embedding

        self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    hparams['n_dims'], hparams['n_heads'],
                    hparams['feedforward_dims'], hparams['dropout']),
                hparams['n_layers'])

        if n_class is None or n_class <= 0:
            n_class = len(self.tokens)  # number of classes to decode to
        self.decoder = SeqFeedForward(hparams['n_dims'], n_class,
                        hidden_layers=hparams['n_decode_layers'] - 1, activation_fn=nn.ReLU)
        self.classify_only = classify_only  # whether to decode all positions or only first one

    def forward(self, x):
        # input dims are (batch, seq), embedding adds channel dim to end
        # swap dimensions from (batch, seq, channel) to (seq, batch, channel)
        embedded = self.embedding(x).permute(1, 0, 2)
        # swap dimensions from (seq, batch, channel) to (batch, channels, seq_len)
        latent = self.transformer_encoder(embedded).permute(1, 2, 0)
        if self.classify_only:
            latent = latent[:, :, 0:1]  # take index 0 of seq as target
        predicted = self.decoder(latent).squeeze() # remove seq dim (dim=2)
        return predicted, latent, embedded


class VariantDecoder(SeqBERT):

    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    hparams['n_dims'], hparams['n_heads'],
                    hparams['feedforward_dims'], hparams['dropout']),
                hparams['n_layers'])

    def forward(self, source, target):
        # input dims are (batch, seq), embedding adds channel dim to end
        # swap dimensions from (batch, seq, channel) to (seq, batch, channel)
        embedded_src = self.embedding(source).permute(1, 0, 2)
        latent = self.transformer_encoder(embedded_src)
        # use same embedding for target seq
        embedded_tgt = self.embedding(target).permute(1, 0, 2)
        decoder_latent = self.transformer_decoder(embedded_tgt, latent)
        if self.classify_only:
            latent = latent[:, :, 0:1]
        predicted = self.decoder(decoder_latent.permute(1, 2, 0))
        # swap dimensions from (seq, batch, channel) to (batch, channels, seq_len)
        return predicted, latent.permute(1, 2, 0), embedded_tgt


class SeqBERTLightningModule(LightningModule):

    def __init__(self, model, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.prev_loss = 10000.

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def load_pretrained_encoder(self, source_model):
        self.model.embedding = source_model.embedding
        self.model.transformer_encoder = source_model.transformer_encoder

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # model params
        parser.add_argument('--mode', default='train', type=str)
        parser.add_argument('--n_dims', default=256, type=int)
        parser.add_argument('--n_heads', default=1, type=int)
        parser.add_argument('--n_layers', default=1, type=int)
        parser.add_argument('--n_decode_layers', default=1, type=int)
        parser.add_argument('--feedforward_dims', default=512, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--position_embedding', default='Sinusoidal', type=str)
        # training params
        parser.add_argument('--seq_len', default=1000, type=int)
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        # log params
        parser.add_argument('--print_progress_freq', default=1000, type=int)
        parser.add_argument('--save_checkpoint_freq', default=1000, type=int)
        parser.add_argument('--load_checkpoint_path', default=None, type=str)
        parser.add_argument('--load_pretrained_model', default=None, type=str)
        parser.add_argument('--test_out_file', default='./test-scores.pt', type=str)
        return parser


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

"""
Note: do not run on multi-GPU (no reduce function defined)
"""
class BinaryPredictTensorMetric(pl.metrics.Metric):

    def __init__(self, dim=0):
        super().__init__(dist_sync_on_step=False)
        self.cat_along_dim = dim
        self.add_state("score", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        score = target - torch.sigmoid(preds)
        score = score.detach().cpu()
        if self.score == []:
            self.score = score
        else:
            self.score = torch.cat([self.score, score], dim=self.cat_along_dim)

    def compute(self):
        return self.score


class PrintGradients(pl.Callback):
    def __init__(self):
        print('zero grad callback loaded')

    def on_before_zero_grad(self, *args, **kwargs):
        print('zero grad callback')
        print(args, kwargs)


def main(ModuleClass):
    parent_parser = ArgumentParser(add_help=False)
    parser = ModuleClass.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1)
    args = parser.parse_args()

    seed_everything(0)
    # defaults
    print(vars(args))
    if args.load_checkpoint_path is not None:
        model = ModuleClass.load_from_checkpoint(args.load_checkpoint_path, **vars(args))
    elif args.load_pretrained_model is not None:
        model = ModuleClass(**vars(args))
        pretrained = ModuleClass.load_from_checkpoint(args.load_pretrained_model)
        pretrained.load_pretrained_encoder( model)
    else:
        model = ModuleClass(**vars(args))
    args.callbacks = [
        CheckpointEveryNSteps(args.save_checkpoint_freq),
        # PrintGradients(),
        ]
    if args.gpus > 0:
        args.callbacks.append(GPUStatsMonitor())
    trainer = Trainer.from_argparse_args(args)
    if args.mode == 'train':
        trainer.fit(model)
    elif args.mode == 'test':
        trainer.test(model)
