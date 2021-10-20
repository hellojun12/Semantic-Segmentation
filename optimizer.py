import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def get_optimizer(model, optimizer_name, lr=1e-3, momentum=0.9, weight_decay=5e-4):
    if optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(scheduler_name, optimizer, lr_decay_step, gamma=0.5):
    if scheduler_name.lower() == 'steplr':
        return StepLR(optimizer, lr_decay_step, gamma)

