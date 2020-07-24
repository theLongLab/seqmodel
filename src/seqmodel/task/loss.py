import torch
import torch.nn as nn

class WeightedLoss(nn.Module):

    def __init__(self, loss_fn_and_weights):
        super().__init__()
        self.loss_fn_and_weights = {}
        for fn, weight in loss_fn_and_weights.items():
            if weight > 0:
                self.loss_fn_and_weights[fn] = torch.tensor(weight)

    def forward(self, reconstructed, target, latent):
        loss = torch.tensor(0.)
        for fn, weight in self.loss_fn_and_weights.items():
            loss += weight * fn(reconstructed, target, latent)
        return loss


# neighbour distance is distance or cosine distance between latent variables at adjacent positions
class NeighbourDistanceLoss(nn.Module):

    def __init__(self, distance_measure=nn.MSELoss()):
        super().__init__()
        self.loss = distance_measure

    def forward(self, reconstructed, target, latent):
        return self.loss(latent[:,:,:-1], latent[:,:, 1:])


class CosineSimilarityLoss(nn.CosineSimilarity):

    def __init__(self, dim=1, eps=1e-08, reducer=torch.sum):
        super().__init__(dim=dim, eps=eps)
        self.reducer = reducer

    def forward(self, x, y):
        return self.reducer(1 - super().forward(x, y))


# wrapper passing inputs, targets
class LambdaLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, reconstructed, target, latent):
        return self.loss_fn(reconstructed, target)
