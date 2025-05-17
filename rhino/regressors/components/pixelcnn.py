import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class DiagonalLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DiagonalLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_i = nn.Conv2d(input_size, hidden_size, kernel_size=1)
        self.W_f = nn.Conv2d(input_size, hidden_size, kernel_size=1)
        self.W_o = nn.Conv2d(input_size, hidden_size, kernel_size=1)
        self.W_g = nn.Conv2d(input_size, hidden_size, kernel_size=1)
        
        self.U_i = MaskedConv2d('B', hidden_size, hidden_size, kernel_size=3, padding=1)
        self.U_f = MaskedConv2d('B', hidden_size, hidden_size, kernel_size=3, padding=1)
        self.U_o = MaskedConv2d('B', hidden_size, hidden_size, kernel_size=3, padding=1)
        self.U_g = MaskedConv2d('B', hidden_size, hidden_size, kernel_size=3, padding=1)
        
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x, state):
        h, c = state
        
        i = torch.sigmoid(self.W_i(x) + self.U_i(h) + self.b_i.view(1, -1, 1, 1))
        f = torch.sigmoid(self.W_f(x) + self.U_f(h) + self.b_f.view(1, -1, 1, 1))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h) + self.b_o.view(1, -1, 1, 1))
        g = torch.tanh(self.W_g(x) + self.U_g(h) + self.b_g.view(1, -1, 1, 1))
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new

class DiagonalBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(DiagonalBiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([DiagonalLSTMCell(
            input_size if i == 0 else hidden_size, 
            hidden_size) for i in range(num_layers)])
        
    def forward(self, x, states=None):
        batch_size, _, height, width = x.size()
        
        if states is None:
            states = [(torch.zeros(batch_size, self.hidden_size, height, width).to(x.device),
                       torch.zeros(batch_size, self.hidden_size, height, width).to(x.device)) 
                      for _ in range(self.num_layers)]
        
        new_states = []
        for i, cell in enumerate(self.cells):
            h, c = states[i]
            new_h, new_c = cell(x if i == 0 else new_h, (h, c))
            new_states.append((new_h, new_c))
        
        return new_h, new_states

class RowLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RowLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_i = nn.Conv2d(input_size, hidden_size, kernel_size=(1, 3), padding=(0, 1))
        self.W_f = nn.Conv2d(input_size, hidden_size, kernel_size=(1, 3), padding=(0, 1))
        self.W_o = nn.Conv2d(input_size, hidden_size, kernel_size=(1, 3), padding=(0, 1))
        self.W_g = nn.Conv2d(input_size, hidden_size, kernel_size=(1, 3), padding=(0, 1))
        
        self.U_i = MaskedConv2d('B', hidden_size, hidden_size, kernel_size=(1, 3), padding=(0, 1))
        self.U_f = MaskedConv2d('B', hidden_size, hidden_size, kernel_size=(1, 3), padding=(0, 1))
        self.U_o = MaskedConv2d('B', hidden_size, hidden_size, kernel_size=(1, 3), padding=(0, 1))
        self.U_g = MaskedConv2d('B', hidden_size, hidden_size, kernel_size=(1, 3), padding=(0, 1))
        
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x, state):
        h, c = state
        
        i = torch.sigmoid(self.W_i(x) + self.U_i(h) + self.b_i.view(1, -1, 1, 1))
        f = torch.sigmoid(self.W_f(x) + self.U_f(h) + self.b_f.view(1, -1, 1, 1))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h) + self.b_o.view(1, -1, 1, 1))
        g = torch.tanh(self.W_g(x) + self.U_g(h) + self.b_g.view(1, -1, 1, 1))
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new

class RowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RowLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([RowLSTMCell(
            input_size if i == 0 else hidden_size, 
            hidden_size) for i in range(num_layers)])
        
    def forward(self, x, states=None):
        batch_size, _, height, width = x.size()
        
        if states is None:
            states = [(torch.zeros(batch_size, self.hidden_size, height, width).to(x.device),
                       torch.zeros(batch_size, self.hidden_size, height, width).to(x.device)) 
                      for _ in range(self.num_layers)]
        
        new_states = []
        for i, cell in enumerate(self.cells):
            h, c = states[i]
            new_h, new_c = cell(x if i == 0 else new_h, (h, c))
            new_states.append((new_h, new_c))
        
        return new_h, new_states

class PixelCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3):
        super(PixelCNNBlock, self).__init__()
        self.conv = MaskedConv2d('A', in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(32, out_channels, eps=1e-5)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x
