import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.autograd import Variable


class Network(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=150, batch_size=32, output_dim=300, num_layers=1, bidirectional=False, img_dim=512):
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=bidirectional)
        
        self.img_linear = nn.Linear(img_dim, hidden_dim)
        self.linear1 = nn.Linear(self.hidden_dim + img_dim, 300)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(300, 300)
        self.softmax = nn.Softmax()
        # self.linear = nn.Linear(self.hidden_dim, output_dim)

    # def init_hidden(hidden):
    #     self.hidden = hidden

    def forward(self, input, img, lengths, y):
        # [input_size, batch_size, hidden_dim]
        img_linear_out = self.img_linear(img)
        # print('image[0] in model:',img[0:6])
        img_linear_out = img_linear_out.unsqueeze(0)
        self.hidden = Variable(img_linear_out, requires_grad=False)
        # print(img_linear_out)

        # lengths, sorted_indices = torch.sort(lengths, descending=True)
        # sorted_indices = sorted_indices.to(input.device)
        # input = input.index_select(1, sorted_indices)
        # print(input.size())
        # print('hidden:', self.hidden)
        input = pack_padded_sequence(input, lengths, enforce_sorted=False)
        lstm_out, self.hidden = self.rnn(input, self.hidden)
        
        lstm_out, output_lengths = pad_packed_sequence(lstm_out)
        # print(output_lengths)
        # print(lstm_out.size())
        lstm_out = lstm_out[:, :, :self.hidden_dim]
        
        # lstm_out = torch.stack([lstm_out[item][i] for i, item in enumerate(output_lengths - 1)])
        # print('lstm_out:', lstm_out)
        # lstm_out = lstm_out.float()
        # print(output_lengths)
        

        lstm_out = lstm_out[-1].double()

        # lstm_out = lstm_out.transpose(0, 1).contiguous()
        # lstm_out = lstm_out.view(self.batch_size, -1)
        # print(lstm_out.size(), img.size())

        # print(output_lengths, lstm_out.size())
        # print(lstm_out, lstm_out[-1].size())
        

        data = torch.cat((lstm_out, img), dim=1)
        data = self.linear1(data)
        data = self.relu(data)
        data = self.linear2(data)

        # print(y.size(), data.size())

        # print('data:', data)

        data = torch.einsum('ijk, ik -> ij', y, data)
        # print(data.size())
        data = self.softmax(data)
        # print('data:', data[0])
        # y_pred = self.linear(lstm_out[-1])
        return data