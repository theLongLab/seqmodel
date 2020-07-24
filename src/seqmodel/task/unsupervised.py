import sys
sys.path.append('./src')
import torch
import torch.nn as nn

from seqmodel.seq.transform import N_BASE, LambdaModule, one_hot


class GenericTask(nn.Module):

    def __init__(self, encoder, decoder, loss_fn, preprocess=one_hot):
        super().__init__()
        self.preprocess = preprocess
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
    
    def forward(self, x):
        with torch.no_grad():
            inputs = self.preprocess(x)
        latent = self.encoder(inputs)
        predicted = self.decoder(latent)
        return latent, predicted

    def loss(self, x):
        latent, predicted = self(x)
        return self.loss_fn(predicted, x, latent)

    def evaluate(self):
        latent, predicted = self(target)
        return None  #TODO: when designing `train.py`


class PredictMaskedTask(GenericTask):

    # mask values: remove from loss (implicit), remove from input, randomly change input, no change predict
    def __init__(self, encoder, decoder, loss_fn, mask_prop=0., random_prop=0., keep_prop=0.
                , mask_value=0):  # need mask_value=float('-inf') for transformer (softmax)
        no_loss_prop = 1. - mask_prop - random_prop - keep_prop
        assert no_loss_prop >= 0. and mask_prop >= 0. and random_prop >= 0. and keep_prop >= 0.

        preprocess = LambdaModule(self.generate_mask, self.randomize_input, one_hot, self.mask_input)
        super().__init__(encoder, decoder, loss_fn, preprocess=preprocess)

        self.mask_props = [no_loss_prop, mask_prop, random_prop, keep_prop]
        self.mask_value = mask_value

        self.NO_LOSS_INDEX = 0
        self.MASK_INDEX = 1
        self.RANDOM_INDEX = 2

    # generate from index vector
    def generate_mask(self, x):
        #TODO need to guarantee minimum number of non-removed positions to avoid nan loss
        self.mask = torch.multinomial(torch.tensor(self.mask_props),
                                    x.nelement(), replacement=True).reshape(x.shape)
        return x

    # apply to index vector
    def randomize_input(self, x):
        return x.masked_scatter(self.mask == self.RANDOM_INDEX, torch.randint(N_BASE, x.shape))

    # apply to one-hot vector
    def mask_input(self, x):
        return x.permute(1, 0, 2).masked_fill(
            (self.mask == self.MASK_INDEX), self.mask_value).permute(1, 0, 2)

    def loss(self, x):
        latent, predicted = self(x)
        target_mask = self.mask != self.NO_LOSS_INDEX
        predicted = predicted.permute(1, 0, 2).masked_select(target_mask).reshape(-1, 4)
        target = x.masked_select(target_mask)
        return self.loss_fn(predicted, target, latent)


class ReconstructionTask(GenericTask):

    def __init__(self):
        pass

    def loss(self, inputs, targets):
        pass


class PredictNextTokenTask(GenericTask):

    def __init__(self):
        pass

    def loss(self, inputs, targets):
        pass
