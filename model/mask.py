import torch

"""
mask 할 곳은 1 padding position: 0
"""
def get_attn_pad_mask(inputs, input_lengths, expand_length):

    def get_transformer_non_pad_mask(inputs, input_lengths):
        batch_size = inputs.size(0)

        if len(inputs.size()) == 2:
            non_pad_mask = inputs.new_ones(inputs.size()) # B x T
        elif len(inputs.size()) == 3:
            non_pad_mask = inputs.new_ones(inputs.size()[:-1]) # B x T
        else:
            raise ValueError("Input Shape Error")

        for i in range(batch_size):
            non_pad_mask[i, input_lengths[i]:] = 0

        return non_pad_mask

    non_pad_mask = get_transformer_non_pad_mask(inputs, input_lengths)
    pad_mask = non_pad_mask.lt(1)
    # torch.lt(input, other, *, out=None) → Tensor
    # 즉 1보다 작은 값들은 True 1보다 크거나 같으면 False
    attn_pad_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    # repeat와 expand 차이점: 결과는 같음
    # 하지만 expand 경우 expand 하고 싶은 dim 이 1이여야하고 장점으로는 추가 memory 사용 X
    # repeat 경우 1 이상일 때 사용해야 하고 추가 메모리 사용

    return attn_pad_mask


