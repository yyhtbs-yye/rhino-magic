import torch.nn as nn

class TwoLayerMLP(nn.Module):

    def __init__(self, in_features, 
                 mid_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU):

        super().__init__()
        out_features = out_features or in_features
        mid_features = mid_features or in_features
        self.fc1 = nn.Linear(in_features, mid_features)
        self.fc2 = nn.Linear(mid_features, out_features)
        self.act = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x