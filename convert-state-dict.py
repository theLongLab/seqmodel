import torch
from collections import OrderedDict

ckpt_fname = 'fname'
checkpoint = torch.load(ckpt_fname + '.ckpt')
state_dict = checkpoint['state_dict']
new_dict = OrderedDict(('model.' + k if k , v for k, v in state_dict))
checkpoint['state_dict'] = new_dict
torch.save(checkpoint, 'fname-fixed.ckpt')