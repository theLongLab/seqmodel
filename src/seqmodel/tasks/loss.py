import torch
import torch.nn as nn

class WeightedLoss(nn.Module):

    def __init__(self, loss_fn_and_weights):
        self.loss_fn_and_weights = {}
        for fn, weight in loss_fn_and_weights.items():
            if weight > 0:
                self.loss_fn_and_weights[fn] = torch.tensor(weight)


    def forward(self, reconstructed, target, latent):
        loss = torch.tensor(0)
        for fn, weight in self.loss_fn_and_weights.items():
            loss += weight * fn(reconstructed, target, latent)


# neighbour distance is distance or cosine distance between latent variables at adjacent positions
class NeighbourDistanceLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, reconstructed, target, latent):
        return self.mse_loss(latent[:,:,:-1], latent[:,:, 1:])


# wrapper
class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self):
        super().__init__()
    
    def forward(self, reconstructed, target, latent):
        return super().forward(reconstructed, target)


# wrapper
class BCELoss(nn.BCELoss):

    def __init__(self):
        super().__init__()
    
    def forward(self, reconstructed, target, latent):
        return super().forward(reconstructed, target)
