import sys
sys.path.append('./src')
import os.path
import torch
import torch.nn as nn
import pytorch_lightning as pl

from seqmodel.model.conv import SeqFeedForward
from seqmodel.model.attention import SinusoidalPosition
from seqmodel.functional.transform import INDEX_TO_BASE


class SeqBERT(nn.Module):

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

    def __init__(self, classify_only=False, **hparams):
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
        self.classify_only = classify_only  # whether to decode all positions or only first one

    def forward(self, x):
        # input dims are (batch, seq), embedding adds channel dim to end
        # swap dimensions from (batch, seq, channel) to (seq, batch, channel)
        embedded = self.embedding(x).permute(1, 0, 2)
        # swap dimensions from (seq, batch, channel) to (batch, channels, seq_len)
        latent = self.transformer(embedded).permute(1, 2, 0)
        if self.classify_only:
            latent = latent[:, :, 0:1]  # take index 0 of seq as target
        predicted = self.decoder(latent).squeeze() # remove seq dim (dim=2)
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
