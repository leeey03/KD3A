# use either ViViT or TimeSformer to extract features 
# model should output clip logits??

# precomputed I3D features from TranSVAE work used as input
import torch
import torch.nn as nn

feature_dict = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048, 
                # adding i3d_trans feature dimension
                "i3d_trans":512}

class I3DTransformerEncoder(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=512, n_layers=2, n_heads=8, data_parallel=True): # check feat_dim
        super(I3DTransformerEncoder, self).__init__()
        
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        if data_parallel:
            self.input_proj = nn.DataParallel(self.input_proj)
            self.transformer = nn.DataParallel(self.transformer)
            self.norm = nn.DataParallel(self.norm)

    def forward(self, x):
        # x: [B, T, D] (I3D features per clip)
        B, T, D = x.shape
        x = self.input_proj(x)   # [B, T, hidden_dim]

        # prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)   # [B, 1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)           # [B, T+1, hidden_dim]

        # transformer expects [S, B, E]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)

        # return CLS embedding
        cls_out = self.norm(x[:, 0])  # [B, hidden_dim]
        return cls_out

class I3DTransformerClassifier(nn.Module):
    def __init__(self, backbone="i3d_trans", classes=8, data_parallel=True):
        super(I3DTransformerClassifier, self).__init__()
        linear = nn.Sequential()
        linear.add_module("fc", nn.Linear(feature_dict[backbone], classes))
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, feature):
        out = self.linear(feature)  # [B, num_classes]
        return out


