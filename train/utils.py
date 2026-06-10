import json
import torch
import numpy as np

def generate_gat_skeleton(regions_json_path):
    """
    Generates a structural base adjacency matrix (skeleton) for GAT,
    mapping anatomical contour lines to their indices in the sorted unique subset.
    Includes self-loops.
    """
    with open(regions_json_path, 'r') as f:
        config = json.load(f)
        
    subsets = config['landmark_subsets']
    
    # Recreate the sorted unique subset indices exactly as in dataset.py
    subset_indices = []
    for sublist in subsets.values():
        subset_indices.extend(sublist)
    subset_indices = sorted(list(set(subset_indices)))
    
    num_nodes = len(subset_indices)
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # Helper to map a landmark index to its position in the subset
    def get_pos(idx):
        return subset_indices.index(idx)
        
    # Connect contours based on their sequence in regions.json
    for name, indices in subsets.items():
        # Connect sequential elements
        for i in range(len(indices) - 1):
            p1 = get_pos(indices[i])
            p2 = get_pos(indices[i+1])
            A[p1, p2] = 1.0
            A[p2, p1] = 1.0
            
        # Connect last to first for closed loops (lips and eyes)
        if 'lips' in name or 'eye' in name:
            p_first = get_pos(indices[0])
            p_last = get_pos(indices[-1])
            A[p_first, p_last] = 1.0
            A[p_last, p_first] = 1.0
            
    # Enforce self-loops explicitly so self-attention is valid
    np.fill_diagonal(A, 1.0)
    
    return torch.from_numpy(A)

def compute_knn_adj(mean_coords, k=5):
    """
    Computes a symmetric k-NN adjacency matrix based on mean coordinates.
    
    Args:
        mean_coords (Tensor): Tensor of shape (num_nodes, 2)
        k (int): Number of nearest neighbors
        
    Returns:
        Tensor: Binary adjacency matrix of shape (num_nodes, num_nodes)
    """
    num_nodes = mean_coords.size(0)
    # Compute pairwise Euclidean distances
    dist_matrix = torch.cdist(mean_coords.unsqueeze(0), mean_coords.unsqueeze(0), p=2).squeeze(0)
    
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i in range(num_nodes):
        # Find k+1 nearest neighbors (including the node itself)
        dists = dist_matrix[i]
        knn_indices = torch.topk(dists, k=k+1, largest=False).indices
        for idx in knn_indices:
            if idx != i:
                adj[i, idx] = 1.0
                adj[idx, i] = 1.0  # Make it symmetric
                
    return adj

def normalize_adjacency(adj):
    """
    Computes the normalized adjacency matrix: D^-1/2 * (A + I) * D^-1/2
    as per Kipf & Welling GCN paper.
    
    Args:
        adj (Tensor): Adjacency matrix of shape (num_nodes, num_nodes)
        
    Returns:
        Tensor: Normalized adjacency matrix
    """
    num_nodes = adj.size(0)
    I = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
    adj_tilde = adj + I
    
    # Compute degree matrix
    deg = torch.sum(adj_tilde, dim=1)
    
    # Compute D^-1/2
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    
    # Return D^-1/2 * A_tilde * D^-1/2
    return torch.matmul(torch.matmul(D_inv_sqrt, adj_tilde), D_inv_sqrt)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Multi-class Focal Loss.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        
        Args:
            alpha (Tensor): Weight tensor of shape (num_classes,)
            gamma (float): Focusing parameter
            reduction (str): 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        import torch.nn.functional as F
        # inputs shape: (batch_size, num_classes)
        # targets shape: (batch_size,)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class prediction
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
