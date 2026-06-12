import torch
import torch.nn as nn
import torch.nn.functional as F

class HighCapacityAdaptiveGATLayer(nn.Module):
    def __init__(self, features, num_nodes, dropout=0.15, alpha=0.2):
        super().__init__()
        self.features = features
        self.dropout = dropout
        self.alpha = alpha
        self.num_nodes = num_nodes

        self.num_heads = 4
        self.head_dim = features // self.num_heads  # 64 channels per head
        
        # 4 independent projection layers (one per head)
        self.W_heads = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in range(self.num_heads)
        ])
        
        # 4 independent learnable attention weight vectors 'a'
        self.a_heads = nn.ParameterList([
            nn.Parameter(torch.empty(size=(2 * self.head_dim, 1))) for _ in range(self.num_heads)
        ])
        for a in self.a_heads:
            nn.init.xavier_uniform_(a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        # 4 independent learnable spatial biases
        self.spatial_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(num_nodes, num_nodes)) for _ in range(self.num_heads)
        ])

    def forward(self, h, return_attention=False):
        # h shape: (batch_size, num_nodes, features)
        batch_size, num_nodes, _ = h.size()
        
        # Split features into 4 heads along the last dimension
        chunks = torch.split(h, self.head_dim, dim=-1)
        
        out_chunks = []
        attn_matrices = []
        
        for i in range(self.num_heads):
            h_h = chunks[i]  # (B, N, head_dim)
            Wh = self.W_heads[i](h_h)  # (B, N, head_dim)
            
            a = self.a_heads[i]
            a_1 = a[:self.head_dim, :]
            a_2 = a[self.head_dim:, :]
            
            f_1 = torch.matmul(Wh, a_1)  # (B, N, 1)
            f_2 = torch.matmul(Wh, a_2)  # (B, N, 1)
            
            # Pairwise attention energies
            energy = self.leakyrelu(f_1 + f_2.transpose(1, 2))  # (B, N, N)
            
            # Normalization with head-specific Learnable Spatial Bias
            attention = F.softmax(energy + self.spatial_biases[i], dim=-1)
            attention_dropped = F.dropout(attention, self.dropout, training=self.training)
            
            h_prime = torch.matmul(attention_dropped, Wh)  # (B, N, head_dim)
            out_chunks.append(h_prime)
            attn_matrices.append(attention)
            
        # Concatenate outputs back to total features dimension
        out = torch.cat(out_chunks, dim=-1)  # (B, N, features)
        
        if return_attention:
            # Average attention across the 4 heads for export/visualization
            attn_avg = torch.mean(torch.stack(attn_matrices, dim=0), dim=0)
            return out, attn_avg
        return out


class GATBackbone(nn.Module):
    def __init__(self, num_nodes=88, hidden_dim=256, num_classes=7, depth=4):
        super().__init__()
        self.depth = depth
        self.num_nodes = num_nodes
        
        # Initial projection from spatial coordinate/distance space (6D) to hidden manifold space
        self.input_projection = nn.Linear(6, hidden_dim)
        
        # Stacked high-capacity adaptive attention layers
        self.gat_layers = nn.ModuleList([
            HighCapacityAdaptiveGATLayer(features=hidden_dim, num_nodes=num_nodes) for _ in range(depth)
        ])
        
        # Post-pooling classification projection layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x input tensor dimension: (batch_size, num_nodes, 6)
        if x.size(1) > self.num_nodes:
            x = x[:, :self.num_nodes, :]
        
        # Map raw dimensions to hidden scale
        h = self.relu(self.input_projection(x))
        
        # Deep execution through attention blocks via residual highways
        last_attention = None
        for i in range(self.depth):
            # Compute dense layer attention, execute dropout, and insert skip connection
            if i == self.depth - 1:
                h_attn, last_attention = self.gat_layers[i](h, return_attention=True)
            else:
                res = self.gat_layers[i](h, return_attention=False)
                if isinstance(res, tuple):
                    h_attn, _ = res
                else:
                    h_attn = res
            h = self.relu(h_attn) + h  # Skip connection breaking over-smoothing loops
            
        # Global Average Pooling (Aggregating cross-node features into single vector)
        g_features = torch.mean(h, dim=1)  # (B, hidden_dim)
        
        # Feed-forward classification block
        out = self.relu(self.fc1(g_features))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits, last_attention

    def extract_features(self, x, return_attention=False):
        # x input tensor dimension: (batch_size, num_nodes, 6)
        if x.size(1) > self.num_nodes:
            x = x[:, :self.num_nodes, :]
        
        # Map raw dimensions to hidden scale
        h = self.relu(self.input_projection(x))
        
        # Deep execution through GAT layers
        last_attention = None
        for i in range(self.depth):
            if return_attention and i == self.depth - 1:
                h_attn, last_attention = self.gat_layers[i](h, return_attention=True)
            else:
                res = self.gat_layers[i](h, return_attention=False)
                h_attn = res[0] if isinstance(res, tuple) else res
            h = self.relu(h_attn) + h
            
        # Global Average Pooling to output a 256-dimensional feature vector
        g_features = torch.mean(h, dim=1)  # (B, hidden_dim)
        if return_attention:
            return g_features, last_attention
        return g_features


class DeepFacialGAT(nn.Module):
    def __init__(self, num_nodes=88, hidden_dim=256, num_classes=7, depth=4):
        super().__init__()
        import torchvision.models as models
        self.gat = GATBackbone(num_nodes=num_nodes, hidden_dim=hidden_dim, num_classes=num_classes, depth=depth)
        
        # Load texture backbone with pretrained weights
        self.texture_net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Fusion Classifier Head: 256 (GAT features) + 960 (CNN features) = 1216
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 960, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, coords, img):
        # 1. Extract GAT features and attention matrix
        gat_feats, attention = self.gat.extract_features(coords, return_attention=True) # (B, 256)
        
        # 2. Extract visual texture features
        cnn_feats = self.texture_net.features(img) # (B, 960, 7, 7)
        cnn_feats = self.pool(cnn_feats) # (B, 960, 1, 1)
        cnn_feats = torch.flatten(cnn_feats, 1) # (B, 960)
        
        # 3. Concatenate and pass to fusion classifier
        fused = torch.cat([gat_feats, cnn_feats], dim=-1) # (B, 1216)
        logits = self.classifier(fused)
        
        return logits, attention


