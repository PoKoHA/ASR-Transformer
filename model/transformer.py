import torch
import torch.nn as nn

from attention import MultiHeadAttention
from VGGExtractor import VGGExtractor
from position import PositionalEncoding
from mask import *
from modules import LayerNorm
from sublayers import *
from utils import init_weight

################################
# Encoder
################################

class EncoderLayer(nn.Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super(EncoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, n_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff), d_model)

    def forward(self, inputs, mask=None):
        attn_output, attn_map = self.self_attention(inputs, inputs, inputs, mask)
        output = self.feed_forward(attn_output)

        return output, attn_map

class Encoder(nn.Module):

    def __init__(self, d_model=512, input_dim=80, d_ff=2048, n_layers=6, n_heads=8, pad_id=0, dropout_p=0.3):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pad_id = pad_id

        self.conv = VGGExtractor(input_dim)
        self.linear = nn.Linear(self.conv.get_output_dim(), d_model) # README 참고 Linear 한번 해주고 나서 Encoder layer 실행
        init_weight(self.linear)

        self.dropout = nn.Dropout(dropout_p)
        self.positional_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, inputs, inputs_lengths):
        conv_outputs, output_lengths = self.conv(inputs, inputs_lengths)
        # output_lengths: maskCNN 을 걸친 후에 줄어드는 length 값

        encoder_mask = get_attn_pad_mask(conv_outputs, output_lengths, conv_outputs.size(1))

        linear = self.linear(conv_outputs)
        pos_encoding = self.positional_encoding(linear.size(1))
        dropout = self.dropout(pos_encoding)

        for layer in self.layers:
            outputs, attn = layer(dropout, encoder_mask)

        return outputs, output_lengths


################################
# Decoder
################################



