import torch
from torch.utils.data import Dataset
import json

class FacialLandmarkDataset(Dataset):
    def __init__(self, data_path, regions_json_path):
        with open(regions_json_path) as f:
            self.regions = json.load(f)
        
        # Flatten subset indices and remove duplicates to get unique nodes
        subset_indices = []
        for sublist in self.regions['landmark_subsets'].values():
            subset_indices.extend(sublist)
        self.subset_indices = sorted(list(set(subset_indices)))
        
        # Log number of selected nodes
        print(f"Loaded dataset subset with {len(self.subset_indices)} unique landmarks out of 468.")
        
        self.data = torch.load(data_path) # Assumes dict: {'coords': Tensor, 'labels': Tensor}

    def __len__(self):
        return len(self.data['labels'])
    
    def __getitem__(self, idx):
        # Filter raw 468/478 nodes to the selected subset nodes
        coords = self.data['coords'][idx][self.subset_indices]
        
        # Normalize relative to anchor index in the subset
        anchor_idx = self.regions['normalization_anchor_index']
        # Find where the anchor is in the subset indices list
        if anchor_idx in self.subset_indices:
            anchor_pos = self.subset_indices.index(anchor_idx)
        else:
            # Fallback to index 1 or 0 if anchor not found
            anchor_pos = 1 if len(self.subset_indices) > 1 else 0
            
        anchor = coords[anchor_pos]
        return coords - anchor, self.data['labels'][idx]
