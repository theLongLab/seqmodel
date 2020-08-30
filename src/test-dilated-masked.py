import torch
import torch.nn as nn
from seqmodel.model.conv import DilateConvEncoder, SeqFeedForward
from seqmodel.task.task import LambdaLoss
from seqmodel.task.mask import PositionMask
from seqmodel.seq.mapseq import MapSequence

encoder = DilateConvEncoder(4, 3, 2, 2., 1, 3, 0.1)
decoder = SeqFeedForward(encoder.out_channels, 4, 1, activation_fn=nn.ReLU)
loss_fn = LambdaLoss(nn.CrossEntropyLoss())
task = PositionMask(encoder, decoder, loss_fn, keep_prop=0.05, mask_prop=0.12, random_prop=0.03)
dataset = MapSequence.from_file('data/ref_genome/chr22.fa', 500, remove_gaps=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
optimizer = torch.optim.Adam(task.parameters(), lr=0.1)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)
task = task.to(device)

best_loss = 999999.
loss_sum = 0.
for i, batch in enumerate(data_loader):
    batch = batch.to(device)
    predicted, target, latent, loss = task.loss(batch)
    loss_sum += loss.item()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 1000 == 0:
        new_loss = loss_sum / len(batch)
        if new_loss < best_loss:
            best_loss = new_loss
            torch.save(task.state_dict(), 'test-dilated-masked-' + str(i) + '.seqmod')
        print(i, new_loss)
        loss_sum = 0.
