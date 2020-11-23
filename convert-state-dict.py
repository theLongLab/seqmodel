import torch
from collections import OrderedDict
import os

ckpt_fname = os.listdir()[0]
checkpoint = torch.load(ckpt_fname)
state_dict = checkpoint['state_dict']
new_dict = OrderedDict(('model.' + k, v) for k, v in state_dict.items())
checkpoint['state_dict'] = new_dict
torch.save(checkpoint, 'fixed-' + ckpt_fname)