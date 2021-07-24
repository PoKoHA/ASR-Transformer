import random

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from VGGExtractor import VGGExtractor
from position import *
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

        encoder_pad_mask = get_attn_pad_mask(conv_outputs, output_lengths, conv_outputs.size(1))

        linear = self.linear(conv_outputs)
        pos_encoding = self.positional_encoding(linear.size(1))
        dropout = self.dropout(pos_encoding)

        for layer in self.layers:
            outputs, attn = layer(dropout, encoder_pad_mask)

        return outputs, output_lengths


################################
# Decoder
################################
class DecoderLayer(nn.Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super(DecoderLayer, self).__init__()

        self.self_attention = AddNorm(MultiHeadAttention(d_model, n_heads), d_model)
        self.cross_attention = AddNorm(MultiHeadAttention(d_model, n_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff), d_model)

    def forward(self, inputs, encoder_outputs, mask=None, cross_mask=None):
        output, self_attn = self.self_attention(inputs, inputs, inputs, mask)
        output, cross_attn = self.cross_attention(output, encoder_outputs, encoder_outputs, cross_mask)
        output = self.feed_forward(output)

        return output, self_attn, cross_attn

class Decoder(nn.Module):

    def __init__(
            self,
            num_classes,
            d_model=512,
            d_ff=2048,
            n_layers=6,
            n_heads=8,
            dropout_p=0.3,
            pad_id=0,
            sos_id=2001,
            eos_id=2002
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.embedding = Embedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout_p)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        self.layerNorm = LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes, bias=False)
        init_weight(self.fc)

    def forward_step(
            self,
            decoder_inputs,
            decoder_inputs_lengths,
            encoder_outputs,
            encoder_outputs_lengths,
            positional_encoding_length
    ):
        decoder_pad_mask = get_attn_pad_mask(
            decoder_inputs, decoder_inputs_lengths, decoder_inputs_lengths.size(1)
        )

        decoder_regression_mask = get_attn_subsequent_mask(decoder_inputs)
        decoder_mask = torch.gt((decoder_pad_mask + decoder_regression_mask), 0)
        # gt 는 lt와 반대 lt: input < other / gt: input > output
        # 즉 0 이랑 같거나 작으면 False

        encoder_pad_mask = get_attn_pad_mask(
            encoder_outputs, encoder_outputs_lengths, decoder_inputs.size(1)
        )

        embedding = self.embedding(decoder_inputs) + self.positional_encoding(positional_encoding_length)
        outputs = self.dropout(embedding)

        for layer in self.layers:
            outputs, self_attn, cross_attn = layer(
                outputs, encoder_outputs, decoder_mask, encoder_pad_mask
            )

        return outputs

    def forward(self, encoder_outputs, targets=None, encoder_outputs_lengths=None, target_lengths=None, teacher_forcing_p=1.0):
        batch_size = encoder_outputs.size(0)
        use_teacher_forcing = True if random.random() < teacher_forcing_p else False

        if targets is not None and use_teacher_forcing:
            targets = targets[targets != self.eos_id].view(batch_size, -1) # eos_id 제외
            target_length = targets.size(1) # eos 제외한 real length

            outputs = self.forward_step(
                decoder_inputs=targets,
                decoder_inputs_lengths=target_lengths,
                encoder_outputs=encoder_outputs,
                encoder_outputs_lengths=encoder_outputs_lengths,
                positional_encoding_length=target_length # 딱 여기까지만 pos encoding 함
            )

            return self.fc(outputs).log_softmax(dim=-1)

        # inference 할 때도 사용
        else:
            logits = list()

            input_var = encoder_outputs.new_zeros(batch_size, self.max_length).long()
            # todo max_length??s
            input_var = input_var.fill_(self.pad_id)
            input_var[:, 0] = self.sos_id

            for di in range(1, self.max_length):
                input_lengths = torch.IntTensor(batch_size).fill_(di)

                outputs = self.forward_step(
                    decoder_inputs=input_var[:, :di],
                    decoder_inputs_lengths=input_lengths,
                    encoder_outputs=encoder_outputs,
                    encoder_outputs_lengths=encoder_outputs_lengths,
                    positional_encoding_length=di
                )

                step_output = self.fc(outputs).log_softmax(dim=-1)

                logits.append(step_output[:, -1, :])
                input_var = logits[-1].topk(1)[1]

            return torch.stack(logits, dim=1)





