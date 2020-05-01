
real_token_len = 181
pad_token_label_id = -100
padding_length = 331
tag_seq = [0] * real_token_len + ([pad_token_label_id] * padding_length)
start_position = 6
end_position = 12
tag_seq[start_position] = 1
tag_seq[start_position+1:end_position+1] = [2]*(end_position-start_position)
print(tag_seq)

import torch
attention_mask = torch.tensor([[1,1,1,0,0],[1,1,0,0,0]])
active_loss = attention_mask.view(-1) == 1
print(active_loss)