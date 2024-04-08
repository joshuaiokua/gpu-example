import torch

def set_device(force_gpu=False):
    if force_gpu:
        print("GPU is forced to be used. Device set to GPU")
        device = torch.device('cuda')

        if not torch.cuda.is_available():
            raise ValueError("GPU is not available. Please set force_gpu=False to use CPU")
        
    if torch.cuda.is_available():
        print("GPU is available. Device set to GPU")
        device = torch.device('cuda')
    else:
        print("GPU is not available. Device set to CPU")
        device = torch.device('cpu')
    return device