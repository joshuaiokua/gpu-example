import torch

def set_device():
    if torch.cuda.is_available():
        print("GPU is available. Device set to GPU")
        device = torch.device('cuda')
    else:
        print("GPU is not available. Device set to CPU")
        device = torch.device('cpu')
    return device