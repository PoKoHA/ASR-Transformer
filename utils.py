import torch
import torch.nn as nn

def init_weight(m):
    nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)






