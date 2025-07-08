import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXt(nn.Module):
    """
    ConvNeXt model with a projection head.

    Args:
        embedding_dim (int): projector embedding dimension.
        pretrained (bool): if True, uses ImageNet pretrained weights.
        use_norm (bool): if True, normalize the embeddings.
    """

    def __init__(self, embedding_dim: int, pretrained: bool = True, use_norm: bool = True):
        super(ConvNeXt, self).__init__()

        self.pretrained = pretrained
        self.use_norm = use_norm
        self.embedding_dim = embedding_dim

        if self.pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = convnext_tiny(weights=weights)

        self.feat_dim = 768
        self.model.classifier = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.embedding_dim)
        )

    def forward(self, x):
        f = self.model(x)

        f = torch.flatten(f, start_dim=1)

        if self.use_norm:
            f = F.normalize(f, dim=1)

        g = self.projector(f)

        if self.use_norm:
            return f, F.normalize(g, dim=1)
        else:
            return f, g
