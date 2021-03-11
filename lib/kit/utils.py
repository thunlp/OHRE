import torch


def cudafy(module):
    if torch.cuda.is_available():
        return module.cuda()
    else:
        return module.cpu()


