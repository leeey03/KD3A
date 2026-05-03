import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


class TemporalDownsampler(nn.Module):
    """Downsamples temporal dimension by taking every nth frame."""
    def __init__(self, downsample_factor: int):
        super().__init__()
        self.downsample_factor = downsample_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x[:, ::self.downsample_factor, :]
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return self.dropout(self.pe[:, :x.size(1), :])

class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with self-attention and FFN."""
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, 
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, 
                                               batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention block
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        
        # Feed-forward block
        x_norm = self.norm2(x)
        ff_out = self.linear2(self.dropout2(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(ff_out)
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""
    def __init__(self, d_model: int, n_heads: int, num_layers: int, 
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class CrossAttentionModule(nn.Module):
    """Cross-attention from slower scale (context) to faster scale (query)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        # query: faster scale [B, T_fast, D]
        # key, value: slower scale [B, T_slow, D]
        query_norm = self.norm(query)
        attn_out, _ = self.cross_attn(query_norm, key, value)
        return query + self.dropout(attn_out)


class ClassificationHead(nn.Module):
    """Classification head for per-scale predictions."""
    def __init__(self, d_model: int, classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.global_pool(x).squeeze(-1)  # [B, D]
        x = self.dropout(x)
        logits = self.fc(x)  # [B, classes]
        return logits


class FusionHead(nn.Module):
    """Ensemble fusion head that concatenates scale features before classification."""
    def __init__(self, d_model: int, classes: int, scales: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Global average pooling to aggregate temporal dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Classification layer for concatenated features (3 scales * d_model)
        self.fc = nn.Linear(d_model * scales, classes)
    
    def forward(self, feat_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from multiple scales via concatenation and classify.
        
        Args:
            feat_list: List of feature tensors from each scale [B, T_i, D]
        
        Returns:
            Fused logits [B, classes]
        """
        # Global average pool each scale's features to [B, D]
        pooled_feats = []
        for feat in feat_list:
            # feat: [B, T_i, D]
            feat_norm = self.norm(feat)
            feat_transposed = feat_norm.transpose(1, 2)  # [B, D, T_i]
            pooled = self.global_pool(feat_transposed).squeeze(-1)  # [B, D]
            pooled_feats.append(pooled)
        
        # Concatenate the pooled features across scales
        fused_feats = torch.cat(pooled_feats, dim=1)  # [B, D*3]
        
        # Classification
        fused_feats = self.dropout(fused_feats)
        logits = self.fc(fused_feats)  # [B, classes]
        
        return logits
class PerScaleTemporalTransformer(nn.Module):
    """
    Temporal transformer processing a single temporal scale. Supports downsampling for usage across different scales.
    """
    def __init__(self, 
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 input_dim: int = 2048,
                 downsample_factor: Optional[int] = None):
        super().__init__()
        self.name = "PerScaleTemporalTransformer"
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Sinusoidal or learnable positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)

        # Temporal downsampler if specified
        self.downsampler = None
        if downsample_factor is not None and downsample_factor > 1:
            self.downsampler = TemporalDownsampler(downsample_factor)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(d_model, n_heads, 
                                          n_layers, 
                                          dim_feedforward, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [B, T, input_dim]
        
        Returns:
            Encoded features [B, T', d_model] where T' depends on downsampling
        """
        B, T, _ = x.shape
        
        # Project input to model dimension
        x = self.input_proj(x)  # [B, T, d_model]
        x = x + self.pos_encoder(x)  # Add position info
        
        # Downsample if applicable
        if self.downsampler is not None:
            x = self.downsampler(x)  # [B, T', d_model]
        
        # Encode with transformer
        feat = self.encoder(x)  # [B, T', d_model]
        
        return feat

class MultiScaleTemporalTransformer(nn.Module):
    """
    Multi-scale temporal transformer for action recognition.
    
    Processes features at three temporal scales (T, T/2, T/4) with:
    - Independent transformer encoders per scale
    - Cross-attention from slower to faster scales
    - Per-scale classification heads
    - Learnable ensemble fusion

    CLS token not used as aggregation is already being done via global average pooling.
    """
    def __init__(self, 
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dim_feedforward: int = 1024,
                 classes: int = 8,
                 dropout: float = 0.1,
                 input_dim: int = 2048):
        super().__init__()
        self.name = "MultiScaleTemporalTransformer"
        
        self.d_model = d_model
        self.classes = classes
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Sinusoidal or learnable positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)

        # Temporal downsamplers (Edited to match late fusion but parameter namings have yet to be updated.)
        self.downsample_2x = TemporalDownsampler(4)
        self.downsample_4x = TemporalDownsampler(16)
        
        # Independent transformer encoders for each scale
        self.encoder_scale1 = TransformerEncoder(d_model, n_heads, 
                                                 n_layers, 
                                                 dim_feedforward, dropout)
        self.encoder_scale2 = TransformerEncoder(d_model, n_heads, 
                                                 n_layers, 
                                                 dim_feedforward, dropout)
        self.encoder_scale4 = TransformerEncoder(d_model, n_heads, 
                                                 n_layers, 
                                                 dim_feedforward, dropout)
        
        # Cross-attention modules (slower → faster)
        self.cross_attn_2to1 = CrossAttentionModule(d_model, n_heads, dropout)
        self.cross_attn_4to2 = CrossAttentionModule(d_model, n_heads, dropout)
        
        # Per-scale classification heads
        self.head_scale1 = ClassificationHead(d_model, classes, dropout)
        self.head_scale2 = ClassificationHead(d_model, classes, dropout)
        self.head_scale4 = ClassificationHead(d_model, classes, dropout)
        
        # Ensemble fusion head
        # hardcoding 3 scales here
        self.fusion_head = FusionHead(d_model, classes, 3, dropout)
    
    def forward(self, x: torch.Tensor, 
                return_all_scales: bool = True) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [B, T, input_dim]
            return_all_scales: If True, return dict with all scale outputs
        
        Returns:
            Final fused logits [B, classes] or dict with per-scale outputs
        """
        B, T, _ = x.shape
        
        # Project input to model dimension
        x = self.input_proj(x)  # [B, T, d_model]
        x = x + self.pos_encoder(x)  # Add position info
        
        # Scale 1: Full temporal resolution
        feat_scale1 = self.encoder_scale1(x)  # [B, T, d_model]
        
        # Scale 2: T/2 temporal resolution
        x_scale2 = self.downsample_2x(x)  # [B, T/2, d_model]
        feat_scale2 = self.encoder_scale2(x_scale2)  # [B, T/2, d_model]
        
        # Scale 4: T/4 temporal resolution
        x_scale4 = self.downsample_4x(x)  # [B, T/4, d_model]
        feat_scale4 = self.encoder_scale4(x_scale4)  # [B, T/4, d_model]
        
        # Cross-attention: Scale 2 → Scale 1
        # Scale 2 acts as context for Scale 1
        feat_scale1 = self.cross_attn_2to1(
            query=feat_scale1,
            key=feat_scale2,
            value=feat_scale2
        )  # [B, T, d_model]
        
        # Cross-attention: Scale 4 → Scale 2
        # Scale 4 acts as context for Scale 2
        feat_scale2 = self.cross_attn_4to2(
            query=feat_scale2,
            key=feat_scale4,
            value=feat_scale4
        )  # [B, T/2, d_model]
        
        # Per-scale classification
        logits_scale1 = self.head_scale1(feat_scale1)  # [B, classes]
        logits_scale2 = self.head_scale2(feat_scale2)  # [B, classes]
        logits_scale4 = self.head_scale4(feat_scale4)  # [B, classes]
        
        # Ensemble fusion of features 
        logits_final = self.fusion_head([feat_scale1, feat_scale2, feat_scale4])
        
        if return_all_scales:
            return {
                'final': logits_final,
                'scale1': logits_scale1,
                'scale2': logits_scale2,
                'scale4': logits_scale4,
                'feat_scale1': feat_scale1,
                'feat_scale2': feat_scale2,
                'feat_scale4': feat_scale4,
            }
        
        return logits_final


# Example usage and training
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 8
    seq_length = 64
    input_dim = 2048
    classes = 8
    
    # Create model
    model = MultiScaleTemporalTransformer(
        d_model=256,
        n_heads=8,
        n_layers=4,
        dim_feedforward=1024,
        classes=classes,
        dropout=0.1,
        input_dim=input_dim
    )
    
    # Example input
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass with all scale outputs
    outputs = model(x, return_all_scales=True)
    
    print(f"Final prediction shape: {outputs['final'].shape}")
    print(f"Scale 1 logits shape: {outputs['scale1'].shape}")
    print(f"Scale 2 logits shape: {outputs['scale2'].shape}")
    print(f"Scale 4 logits shape: {outputs['scale4'].shape}")
    print(f"Scale 1 features shape: {outputs['feat_scale1'].shape}")
    print(f"Scale 2 features shape: {outputs['feat_scale2'].shape}")
    print(f"Scale 4 features shape: {outputs['feat_scale4'].shape}")
    
    # Loss computation with auxiliary losses
    target = torch.randint(0, classes, (batch_size,))
    criterion = nn.CrossEntropyLoss()
    
    # Total loss with auxiliary scale losses
    loss_main = criterion(outputs['final'], target)
    loss_scale1 = criterion(outputs['scale1'], target)
    loss_scale2 = criterion(outputs['scale2'], target)
    loss_scale4 = criterion(outputs['scale4'], target)
    
    # Weighted combination (you can adjust weights)
    total_loss = loss_main + 0.3 * (loss_scale1 + loss_scale2 + loss_scale4)
    
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"Main loss: {loss_main.item():.4f}")
    print(f"Scale losses: {loss_scale1.item():.4f}, {loss_scale2.item():.4f}, {loss_scale4.item():.4f}")