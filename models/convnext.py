import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXt(nn.Module):
    """
    ConvNeXt model with a projection head (baseada em convnext_tiny).

    Args:
        embedding_dim (int): Dimensão da embedding do projector.
        pretrained (bool): Se True, usa pesos pré-treinados no ImageNet.
        use_norm (bool): Se True, normaliza as embeddings.
    """

    def __init__(self, embedding_dim: int, pretrained: bool = True, use_norm: bool = True):
        super(ConvNeXt, self).__init__()

        self.pretrained = pretrained
        self.use_norm = use_norm
        self.embedding_dim = embedding_dim

        # Carrega a ConvNeXt-tiny (versão mais leve)
        if self.pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = convnext_tiny(weights=weights)

        # Remove a camada fully connected original (classificador)
        self.feat_dim = 768  # Dimensão de saída da ConvNeXt-tiny (vs. 2048 da ResNet-50)
        self.model.classifier = nn.Identity()  # Remove o classificador

        # Projetor MLP (similar ao original, mas ajustado para a dimensão da ConvNeXt)
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),  # ConvNeXt usa GELU em vez de ReLU
            nn.Linear(self.feat_dim, self.embedding_dim)
        )

    def forward(self, x):
        # Extrai features
        f = self.model(x)  # Saída já é [batch_size, 768] devido ao nn.Identity()

        f = torch.flatten(f, start_dim=1)

        if self.use_norm:
            f = F.normalize(f, dim=1)

        # Projeta para o espaço latente
        g = self.projector(f)

        if self.use_norm:
            return f, F.normalize(g, dim=1)
        else:
            return f, g


if __name__ == '__main__':
    model = ConvNeXt(embedding_dim=128, pretrained=True)
    x = torch.randn(4, 3, 224, 224)  # Batch de imagens
    features, projections = model(x)

    print(f"Features shape: {features.shape}")  # [4, 768]
    print(f"Projections shape: {projections.shape}")  # [4, 128]
