import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from seqmodel.model.conv import DilateConvEncoder, SeqFeedForward
from seqmodel.task.task import LambdaLoss
from seqmodel.task.unsupervised import PredictMaskedToken
from seqmodel.seq.mapseq import MapSequence


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
        predicted, target, latent, loss = self.task.loss(batch)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}


dataset = MapSequence.from_file('data/ref_genome/chr22.fa', 500, remove_gaps=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)

model = DilatedMasked()
trainer = pl.Trainer()
trainer.fit(model, data_loader)
