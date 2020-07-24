import torch


if __name__ == '__main__':

# booleans and int 8 are identical
dev = torch.device('cuda')
for torch_type in [torch.bool, torch.int8, torch.float16]:
    a = torch.randint(2, (100, 1000, 1000)).type(torch_type).to(dev)
    print(torch.cuda.memory_allocated(), torch_type)
    del a

