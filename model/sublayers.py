import torch
import torch.nn as nn

class PositionWiseFeedForwardNet(nn.Module):

    def __init__(self, d_model=512, d_ff=2048):
        super(PositionWiseFeedForwardNet, self).__init__()

        # Transormer paper) 원래 Linear 2번이지만 Conv1d kernel=1 로하는 것랑 일치
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs.transpose(1, 2))
        relu = self.relu(conv1)
        conv2 = self.conv2(relu)
        output = conv2.transpose(1, 2)

        return output

























