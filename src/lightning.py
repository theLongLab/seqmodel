import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from seqmodel.model.conv import DilateConvEncoder, SeqFeedForward
from seqmodel.task.task import LambdaLoss
from seqmodel.task.unsupervised import PredictMaskedToken
from seqmodel.seq.mapseq import MapSequence
from seqmodel.seq.transform import INDEX_TO_BASE
from seqmodel.task.log import prediction_histograms, normalize_histogram, \
                            summarize, correct, accuracy_per_class


class DilatedMasked(LightningModule):

    def __init__(self):
        super().__init__()
        encoder = DilateConvEncoder(4, 3, 2, 2., 1, 3, 0.1)
        decoder = SeqFeedForward(encoder.out_channels, 4, 1, activation_fn=nn.ReLU)
        loss_fn = LambdaLoss(nn.CrossEntropyLoss())
        self.task = PredictMaskedToken(encoder, decoder, loss_fn, keep_prop=0.05, mask_prop=0.12, random_prop=0.03)

    def configure_optimizers(self):
        return torch.optim.Adam(self.task.parameters(), lr=0.1)

    def forward(self, x):
        return self.task(x)

    def training_step(self, batch, batch_idx):
        predicted, masked_predicted, masked_target, latent, mask, loss = self.task.loss(batch)
        str_train_sample = summarize(mask + 4, batch, correct(predicted, batch),
                predicted.permute(1, 0, 2), index_symbols=INDEX_TO_BASE + [' ', '_', '?', '='])
        hist = prediction_histograms(predicted, batch, n_bins=3)
        acc = normalize_histogram(hist)
        acc_numbers = accuracy_per_class(hist)
        str_acc = summarize(acc, col_labels=INDEX_TO_BASE, normalize_fn=None)
        print(acc_numbers, str_acc, str_train_sample, sep='\n')
        return {'loss': loss,
                'log': {'train_loss': loss}}

    # def validation_step(self, batch, batch_idx):
    #     {'loss': ,
    #     'correct': ,}
    #     result.log('val_loss', loss)

    # def validation_epoch_end(self, val_step_outputs):

    #     result = pl.EvalResult(checkpoint_on=loss)
    #     result.log('val_loss', loss)


train_data = MapSequence.from_file('data/ref_genome/chr22_excerpt_4m.fa', 500, remove_gaps=True)
valid_data = MapSequence.from_file('data/ref_genome/chr22_excerpt_800k.fa', 500, remove_gaps=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=20, shuffle=False, num_workers=4)

model = DilatedMasked()
trainer = pl.Trainer(gpus=0)
trainer.fit(model, train_loader)

