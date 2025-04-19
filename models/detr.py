import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import misc as misc_nn_ops

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, x):
        # Simple 2D sine-cosine positional encoding
        B, C, H, W = x.size()
        device = x.device

        y_embed = torch.linspace(0, 1, H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.linspace(0, 1, W, device=device).unsqueeze(0).repeat(H, 1)

        pos = torch.cat([
            torch.sin(x_embed * 3.1416),
            torch.cos(x_embed * 3.1416),
            torch.sin(y_embed * 3.1416),
            torch.cos(y_embed * 3.1416)
        ], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        return pos

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_queries=100):
        super().__init__()

        # Backbone
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # Positional encoding
        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6
        )

        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))  # object queries

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object" class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        features = self.conv(features)

        B, C, H, W = features.shape
        pos = self.pos_encoder(features)
        src = features.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        pos = pos.flatten(2).permute(2, 0, 1)

        query_pos = self.query_pos.unsqueeze(1).repeat(1, B, 1)

        tgt = torch.zeros_like(query_pos)
        hs = self.transformer(src + pos, tgt + query_pos)

        out = hs[-1]
        outputs_class = self.class_embed(out)
        outputs_coord = self.bbox_embed(out).sigmoid()

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x
