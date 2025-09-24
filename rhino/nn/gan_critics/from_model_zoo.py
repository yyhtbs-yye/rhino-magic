import torch, torch.nn as nn
from torchvision import transforms as T

class BinaryClassifier(nn.Module):
    def __init__(self, backbone_name='resnet18', zoo_name='torchvision',
                 pretrained=False, adaptive_norm=False):
        super().__init__()

        if adaptive_norm:
            self.scale = nn.Parameter(torch.ones(1, 3, 1, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)
        else:
            self.scale = torch.ones(1, 3, 1, 1, requires_grad=False)
            self.bias = torch.zeros(1, 3, 1, 1, requires_grad=False)

        self.backbone, feat_dim = self._init_backbone(zoo_name, backbone_name, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)             # global pooling
        self.head = nn.Linear(feat_dim, 1)

    def _init_backbone(self, zoo_name, backbone_name, pretrained):
        if zoo_name == 'timm':
            import timm
            m = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='')
            return m, m.num_features
        elif zoo_name == 'torchvision':
            import torchvision.models as tvm
            m = getattr(tvm, backbone_name)(pretrained=pretrained)
            for attr in ('fc','classifier','head','heads','last_linear'):
                if hasattr(m, attr): setattr(m, attr, nn.Identity())
            with torch.no_grad():
                y = m(torch.zeros(1,3,224,224))
                feat_dim = y.shape[1] if y.ndim==4 else y.shape[-1]
            return m, feat_dim

    def forward(self, x):
        f = self.backbone(x * self.scale + self.bias)
        if f.ndim==4: 
            f = self.pool(f).flatten(1)
        return torch.sigmoid(self.head(f)).squeeze(1)
