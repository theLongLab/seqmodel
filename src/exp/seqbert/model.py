import sys
sys.path.append('./src')
import os.path
import torch
import torch.nn as nn
import pytorch_lightning as pl

from seqmodel.model.conv import SeqFeedForward
from seqmodel.model.attention import SinusoidalPosition
from exp.seqbert import TOKENS_BP


def bool_to_tokens(bool_tensor, target_tensor_type=torch.long):
    return bool_tensor.to(target_tensor_type) + 7  # 'f' token is 0 + 7, so 1 + 7 = 8 is true


class SeqBERT(nn.Module):

    def __init__(self, **hparams):
        super().__init__()
        self.tokens = self.TOKENS_BP  # may have different tokenizations in the future
        embedding = nn.Embedding(len(self.tokens), hparams['n_dims'])
        if hparams['position_embedding'] == 'Sinusoidal':
            self.embedding = nn.Sequential(
                embedding,
                SinusoidalPosition(hparams['n_dims'], dropout=hparams['dropout'],
                                    max_len=(hparams['seq_len'] + 1)),  # add 1 for cls token
                )
        else:
            self.embedding = embedding

        self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hparams['n_dims'], hparams['n_heads'],
                                    hparams['feedforward_dims'], hparams['dropout']),
                                    hparams['n_layers'])

        n_class = len(self.tokens)  # number of classes to decode to
        if 'n_class' in hparams and hparams['n_class'] > 0:
            n_class = hparams['n_class']
        self.decoder = SeqFeedForward(hparams['n_dims'], n_class,
                        hidden_layers=hparams['n_decode_layers'] - 1, activation_fn=nn.ReLU)

        self.classify_only = False  # whether to decode all positions or only first one
        if ('mode' in hparams) and hparams['mode'] == 'classify':
            self.classify_only = True

    def forward(self, x):
        # input dims are (batch, seq), embedding adds channel dim to end
        # swap dimensions from (batch, seq, channel) to (seq, batch, channel)
        embedded = self.embedding(x).permute(1, 0, 2)
        # swap dimensions from (seq, batch, channel) to (batch, channels, seq_len)
        latent = self.transformer(embedded).permute(1, 2, 0)
        if self.classify_only:
            latent = latent[:, :, 0:1]
        predicted = self.decoder(latent)
        return predicted, latent, embedded


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
