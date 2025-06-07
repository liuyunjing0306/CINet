import torch
import torch.nn as nn
import torch.nn.functional as F


def get_cuda(tensor, gpu):
    # if torch.cuda.is_available():
    #     tensor = tensor
    return tensor.to(torch.device('cuda:{}'.format(gpu)))


class Classifier(nn.Module):
    def __init__(self, gpu):
        super().__init__()
        self.gpu = gpu
        self.fc1 = nn.Linear(10, 10)

    def forward(self, input):
        out = self.fc1(input)

        return out




