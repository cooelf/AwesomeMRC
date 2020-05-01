import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin

class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super().__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim * 2, dim))

        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, control):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memory, know, control):
        mem = self.mem(memory[-1]).unsqueeze(2)
        concat = self.concat(torch.cat([mem * know, know], 1) \
                                .permute(0, 2, 1))
        attn = concat * control[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(1)

        read = (attn * know).sum(2)

        return read


class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super().__init__()

        self.concat = linear(dim * 2, dim)

        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, memories, retrieved, controls):
        prev_mem = memories[-1]
        concat = self.concat(torch.cat([retrieved, prev_mem], 1))
        next_mem = concat

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.mem(attn_mem) + concat

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = F.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem

        return next_mem


class MACUnit(nn.Module):
    def __init__(self, dim, max_step=3,
                self_attention=False, memory_gate=False,
                dropout=0.15, use_one = False):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout
        self.use_one = use_one

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            context_i = context[:,i]
            question_i = question[:,i]
            knowledge_i = knowledge[:,i]
            # print("context question knowledge", context.size(), question.size(), knowledge.size())
            # print(context_i.size(), question_i.size(), knowledge_i.size())
            control = self.control(i, context_i, question_i, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read = self.read(memories, knowledge_i, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

        return memory


class MACNetworkOri(nn.Module):
    def __init__(self, dim, embed_hidden=768,
                max_step=12, self_attention=False, memory_gate=False,
                classes=4, dropout=0.15):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(24, dim, 3, padding=1),
                                nn.ELU(),
                                nn.Conv2d(dim, dim, 3, padding=1),
                                nn.ELU())

        # self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, dim,
                        batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(dim * 2, dim)

        self.mac = MACUnit(dim, max_step,
                        self_attention, memory_gate, dropout)


        self.classifier = nn.Sequential(linear(dim * 3, dim),
                                        nn.ELU())
        # self.classifier = nn.Sequential(linear(dim * 3, dim))

        self.max_step = max_step
        self.dim = dim

        self.reset()

    def reset(self):
        # self.embed.weight.data.uniform_(0, 1)

        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

        kaiming_uniform_(self.classifier[0].weight)

    def forward(self, image, question, question_len=None, srl_ques=None, dropout=0.15):
        b_size = question.size(0)

        # img = self.conv(image)
        img = image.view(b_size, self.dim, -1) # batch * dim * img_len
        # img = image.permute(0, 2, 1)
        # print("img size:", img.size())

        # embed = self.embed(question)
        embed = question  # batch * ques_len * dim
        # print("embed size:", embed.size())
        # embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
        #                                         batch_first=True)
        self.lstm.flatten_parameters()
        lstm_out, (h, _) = self.lstm(embed)  #
        # print("lstm_out size:", lstm_out.size())
        # lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
        #                                             batch_first=True)
        lstm_out = self.lstm_proj(lstm_out)
        # print("h size:", h.size())
        h = h.permute(1, 0, 2).contiguous().view(b_size, -1)
        # print("h size:", h.size())

        memory = self.mac(lstm_out, h, img) #[4,10,768]  [4,1536]  [4,768,10]

        # *************zhou add************
        memory = memory.repeat(image.size(1), 1)
        memory = memory.view(image.size(0), image.size(1), -1)
        # out = torch.cat([memory, h], -1)
        # out = self.classifier(out)

        return out


class MACNetwork(nn.Module):
    def __init__(self, dim, embed_hidden=768,
                max_step=3, self_attention=False, memory_gate=False,
                classes=4, dropout=0.15):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(24, dim, 3, padding=1),
                                nn.ELU(),
                                nn.Conv2d(dim, dim, 3, padding=1),
                                nn.ELU())
        self.lstm = nn.LSTM(embed_hidden, dim,
                        batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(dim * 2, dim)
        self.mac = MACUnit(dim, max_step,
                        self_attention, memory_gate, dropout)
        self.classifier = nn.Sequential(linear(dim * 3, dim),
                                        nn.ELU())
        self.max_step = max_step
        self.dim = dim
        self.reset()

    def reset(self):
        # self.embed.weight.data.uniform_(0, 1)
        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()
        kaiming_uniform_(self.classifier[0].weight)

    def forward(self, image, question, question_len=None, dropout=0.15):
        b_size = question.size(0)
        srl_aspect = question.size(1)
        ques_len = question.size(2)
        hidden_dim = question.size(-1)

        # img = self.conv(image)
        img = image.view(b_size, srl_aspect, self.dim, -1) # batch * dim * img_len
        # img = image.permute(0, 2, 1)
        # img = img[:,0]

        # embed = self.embed(question)
        # embed = question  # batch * ques_len * dim

        srl_lstm_out = question.new(question.size()).zero_()
        srl_h = question.new(question.size(0), question.size(1), 2*question.size(-1))

        flat_question = question.view(-1, question.size(-2), question.size(-1))
        lstm_out, (h, _) = self.lstm(flat_question)
        lstm_out = self.lstm_proj(lstm_out)
        srl_h = h.permute(1, 0, 2).contiguous().view(b_size, srl_aspect, -1)
        srl_lstm_out = lstm_out.view(b_size, srl_aspect, ques_len, -1)

        # for i in range(srl_aspect):
        #     # print(question.size())
        #     embed = question[:,i]
        #     # print(embed.size())
        #     lstm_out, (h, _) = self.lstm(embed)
        #
        #     lstm_out = self.lstm_proj(lstm_out)
        #     h = h.permute(1, 0, 2).contiguous().view(b_size, -1)
        #
        #     srl_lstm_out[:,i] = lstm_out
        #     srl_h[:,i] = h

        # lstm_out, (h, _) = self.lstm(embed)
        # lstm_out = self.lstm_proj(lstm_out)
        # h = h.permute(1, 0, 2).contiguous().view(b_size, -1)
        memory = self.mac(srl_lstm_out, srl_h, img) #[4,10,768]  [4,1536]  [4,768,10]

        # *************MemRepeat************
        # memory = memory.repeat(image.size(1), 1)
        # memory = memory.view(image.size(0), image.size(1), -1)
        out = torch.cat([memory, srl_h[:,-1,:]], -1)
        out = self.classifier(out)

        return out

class MACNetworkQAGRU(nn.Module):
    def __init__(self, dim, embed_hidden=768,
                max_step=3, self_attention=False, memory_gate=False,
                classes=4, dropout=0.15):
        super().__init__()

        self.gru = nn.GRU(embed_hidden, dim,
                        batch_first=True, bidirectional=True)
        self.gru_proj = nn.Linear(dim * 2, dim)
        self.mac = MACUnit(dim, max_step,
                        self_attention, memory_gate, dropout)
        self.classifier = nn.Sequential(linear(dim * 3, dim),
                                        nn.ELU())
        self.max_step = max_step
        self.dim = dim

    def forward(self, image, question, question_len=None, dropout=0.15):
        b_size = question.size(0)
        srl_aspect = question.size(1)
        ques_len = question.size(2)

        # img = self.conv(image)
        img = image.view(b_size, srl_aspect, self.dim, -1) # batch * dim * img_len
        flat_question = question.view(-1, question.size(-2), question.size(-1))

        self.gru.flatten_parameters()
        gru_out, h = self.gru(flat_question)
        gru_out = self.gru_proj(gru_out)
        srl_h = h.permute(1, 0, 2).contiguous().view(b_size, srl_aspect, -1)
        srl_gru_out = gru_out.view(b_size, srl_aspect, ques_len, -1)

        memory = self.mac(srl_gru_out, srl_h, img) #[4,10,768]  [4,1536]  [4,768,10]

        # *************MemRepeat************
        memory = memory.repeat(image.size(2), 1)
        out = memory.view(image.size(0), image.size(2), image.size(-1))

        return out

class MACNetworkQA(nn.Module):
    def __init__(self, dim, embed_hidden=768,
                max_step=3, self_attention=False, memory_gate=False,
                classes=4, dropout=0.15):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(24, dim, 3, padding=1),
                                nn.ELU(),
                                nn.Conv2d(dim, dim, 3, padding=1),
                                nn.ELU())
        self.lstm = nn.LSTM(embed_hidden, dim,
                        batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(dim * 2, dim)
        self.mac = MACUnit(dim, max_step,
                        self_attention, memory_gate, dropout)
        self.classifier = nn.Sequential(linear(dim * 3, dim),
                                        nn.ELU())
        self.max_step = max_step
        self.dim = dim
        self.reset()

    def reset(self):
        # self.embed.weight.data.uniform_(0, 1)
        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()
        kaiming_uniform_(self.classifier[0].weight)

    def forward(self, image, question, question_len=None, dropout=0.15):
        b_size = question.size(0)
        srl_aspect = question.size(1)
        ques_len = question.size(2)
        hidden_dim = question.size(-1)

        # img = self.conv(image)
        img = image.view(b_size, srl_aspect, self.dim, -1) # batch * dim * img_len
        # img = image.permute(0, 2, 1)
        # img = img[:,0]

        # embed = self.embed(question)
        # embed = question  # batch * ques_len * dim

        srl_lstm_out = question.new(question.size()).zero_()
        srl_h = question.new(question.size(0), question.size(1), 2*question.size(-1))

        flat_question = question.view(-1, question.size(-2), question.size(-1))
        lstm_out, (h, _) = self.lstm(flat_question)
        lstm_out = self.lstm_proj(lstm_out)
        srl_h = h.permute(1, 0, 2).contiguous().view(b_size, srl_aspect, -1)
        srl_lstm_out = lstm_out.view(b_size, srl_aspect, ques_len, -1)

        memory = self.mac(srl_lstm_out, srl_h, img) #[4,10,768]  [4,1536]  [4,768,10]

        # *************MemRepeat************
        memory = memory.repeat(image.size(2), 1)
        out = memory.view(image.size(0), image.size(2), image.size(-1))

        return out

if __name__ == '__main__':
    #For span-based SQuAD
    max_step = 1
    seq = torch.Tensor(4, 10, 768)
    seq = seq.unsqueeze(1)
    print(seq.size())
    net = MACNetworkQAGRU(768, max_step=max_step)
    net.train()
    out = net(seq, seq)
    #print(ques.repeat(srl_ques.size(1), 1, 1).view(ques.size(0), -1, ques.size(1), ques.size(2)).size())
    print(out.size())

    #For classification
    # max_step = 1
    # seq1 = torch.Tensor(4, max_step, 10, 768)
    # seq2 = torch.Tensor(4, max_step, 20, 768)
    # net = MACNetwork(768, max_step=max_step)
    # net.train()
    # out = net(seq1, seq2)
    # #print(ques.repeat(srl_ques.size(1), 1, 1).view(ques.size(0), -1, ques.size(1), ques.size(2)).size())
    # print(out.size())