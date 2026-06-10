import torch
import torch.nn as nn
import torch.nn.functional as F

class HighCapacityAdaptiveGATLayer(nn.Module):
    def __init__(self, features, dropout=0.15, alpha=0.2):
        super().__init__()
        self.features = features
        self.dropout = dropout
        self.alpha = alpha

        # Node feature projection matrix W
        self.W = nn.Linear(features, features, bias=False)
        
        # Attention weight vector 'a'
        self.a = nn.Parameter(torch.empty(size=(2 * features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        # h shape: (batch_size, num_nodes, features)
        batch_size, num_nodes, _ = h.size()
        Wh = self.W(h)  # (B, N, F)
        
        # Generate fully-dense global combinations matrix for unmasked cross-talk
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, num_nodes, 1)
        
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        all_combinations_matrix = all_combinations_matrix.view(batch_size, num_nodes, num_nodes, 2 * self.features)
        
        # Compute fully dense, raw pairwise structural energies
        energy = self.leakyrelu(torch.matmul(all_combinations_matrix, self.a).squeeze(-1))
        
        # Softmax normalization across all nodes globally
        attention = F.softmax(energy, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Structural aggregation: (B, N, F)
        h_prime = torch.matmul(attention, Wh)
        return h_prime


class DeepFacialGAT(nn.Module):
    def __init__(self, num_nodes=88, hidden_dim=256, num_classes=7, depth=4):
        super().__init__()
        self.depth = depth
        
        # Initial projection from spatial coordinate space to hidden manifold space
        self.input_projection = nn.Linear(2, hidden_dim)
        
        # Stacked high-capacity adaptive attention layers
        self.gat_layers = nn.ModuleList([
            HighCapacityAdaptiveGATLayer(features=hidden_dim) for _ in range(depth)
        ])
        
        # Post-pooling classification projection layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x input tensor dimension: (batch_size, num_nodes, 2)
        
        # Map raw dimensions to hidden scale
        h = self.relu(self.input_projection(x))
        
        # Deep execution through attention blocks via residual highways
        for i in range(self.depth):
            # Compute dense layer attention, execute dropout, and insert skip connection
            h_attn = self.gat_layers[i](h)
            h = self.relu(h_attn) + h  # Skip connection breaking over-smoothing loops
            
        # Global Average Pooling (Aggregating cross-node features into single vector)
        g_features = torch.mean(h, dim=1)  # (B, hidden_dim)
        
        # Feed-forward classification block
        out = self.relu(self.fc1(g_features))
        out = self.dropout(out)
        return self.fc2(out)
