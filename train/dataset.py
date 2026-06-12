import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms

NEIGHBORS = [
    [11, 51, 12, 52], [2, 3, 4, 5], [1, 3, 4, 5], [4, 2, 1, 5], [3, 5, 2, 42], [42, 4, 3, 2], [48, 41, 10, 30], [21, 61, 20, 60], [23, 63, 44, 83], [22, 62, 45, 84], [48, 40, 6, 39], [0, 12, 13, 51], [13, 11, 46, 0], [46, 12, 14, 11], [46, 32, 13, 25], [17, 27, 16, 28], [27, 28, 15, 17], [15, 28, 27, 16], [47, 26, 24, 19], [47, 20, 18, 21], [19, 21, 47, 7], [20, 7, 19, 61], [45, 9, 25, 62], [44, 8, 24, 63], [26, 44, 18, 23], [32, 45, 14, 22], [24, 18, 47, 44], [15, 16, 17, 28], [16, 17, 27, 15], [35, 43, 34, 36], [41, 31, 6, 33], [33, 30, 41, 34], [14, 25, 46, 45], [34, 31, 35, 30], [35, 33, 29, 31], [29, 34, 43, 33], [43, 37, 29, 38], [38, 36, 43, 39], [37, 39, 40, 36], [40, 38, 10, 37], [39, 10, 48, 38], [6, 30, 48, 31], [5, 4, 3, 2], [29, 36, 35, 37], [24, 23, 26, 8], [25, 22, 32, 9], [14, 13, 32, 12], [18, 19, 26, 20], [10, 6, 40, 41], [87, 81, 50, 70], [87, 80, 49, 79], [0, 52, 53, 11], [53, 51, 85, 0], [85, 52, 54, 51], [85, 72, 53, 65], [57, 67, 56, 68], [67, 68, 55, 57], [55, 68, 67, 56], [86, 66, 64, 59], [86, 60, 58, 61], [59, 61, 86, 7], [60, 7, 59, 21], [84, 9, 65, 22], [83, 8, 64, 23], [66, 83, 58, 63], [72, 84, 54, 62], [64, 58, 86, 83], [55, 56, 57, 68], [56, 57, 67, 55], [75, 82, 74, 76], [81, 71, 49, 73], [73, 70, 81, 74], [54, 65, 85, 84], [74, 71, 75, 70], [75, 73, 69, 71], [69, 74, 82, 73], [82, 77, 69, 78], [78, 76, 82, 79], [77, 79, 80, 76], [80, 78, 50, 77], [79, 50, 87, 78], [49, 70, 87, 71], [69, 76, 75, 77], [64, 63, 66, 8], [65, 62, 72, 9], [54, 53, 72, 52], [58, 59, 66, 60], [50, 49, 80, 81]
]

class FacialLandmarkDataset(Dataset):
    def __init__(self, data_path, regions_json_path, is_train=False):
        self.is_train = is_train
        with open(regions_json_path) as f:
            self.regions = json.load(f)
        
        # Flatten subset indices and remove duplicates to get unique nodes (all 88 nodes)
        subset_indices = []
        for sublist in self.regions['landmark_subsets'].values():
            subset_indices.extend(sublist)
        self.subset_indices = sorted(list(set(subset_indices)))
        
        # Log number of selected nodes
        print(f"Loaded dataset subset with {len(self.subset_indices)} unique landmarks out of 468/478.")
        
        self.data = torch.load(data_path) # Assumes dict: {'coords': Tensor, 'labels': Tensor, 'image_paths': list}
        self.neighbor_tensor = torch.tensor(NEIGHBORS, dtype=torch.long)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data['labels'])
    
    def __getitem__(self, idx):
        # 1. Filter raw nodes to the full 88 nodes for all calculations
        coords = self.data['coords'][idx][self.subset_indices]
        
        # Center-normalize relative to anchor index (Nose Tip)
        anchor_idx = self.regions['normalization_anchor_index']
        if anchor_idx in self.subset_indices:
            anchor_pos = self.subset_indices.index(anchor_idx)
        else:
            anchor_pos = 1 if len(self.subset_indices) > 1 else 0
            
        anchor = coords[anchor_pos]
        centered_coords = coords - anchor
        
        # Scale-normalize relative to Interocular Distance
        left_eye = centered_coords[10] # subset index 10 is MediaPipe 33 (Left Eye)
        right_eye = centered_coords[50] # subset index 50 is MediaPipe 263 (Right Eye)
        interocular_distance = torch.sqrt(torch.sum((left_eye - right_eye) ** 2))
        
        if interocular_distance > 1e-6:
            normalized_coords = centered_coords / interocular_distance
        else:
            normalized_coords = centered_coords
            
        # 2. Compute explicit spatial ratios (MAR and Brow-Eye Clearance) on the normalized coordinates
        # Mouth: Upper lip 13, Lower lip 14, Corners 78, 308
        idx_13 = self.subset_indices.index(13)
        idx_14 = self.subset_indices.index(14)
        idx_78 = self.subset_indices.index(78)
        idx_308 = self.subset_indices.index(308)
        
        vertical_dist = torch.sqrt(torch.sum((normalized_coords[idx_13] - normalized_coords[idx_14]) ** 2))
        horizontal_dist = torch.sqrt(torch.sum((normalized_coords[idx_78] - normalized_coords[idx_308]) ** 2))
        mar = vertical_dist / (horizontal_dist + 1e-6)
        
        # Brow-Eye: left brow/lid (66, 159), right brow/lid (296, 386)
        idx_66 = self.subset_indices.index(66)
        idx_159 = self.subset_indices.index(159)
        idx_296 = self.subset_indices.index(296)
        idx_386 = self.subset_indices.index(386)
        
        left_clearance = torch.sqrt(torch.sum((normalized_coords[idx_66] - normalized_coords[idx_159]) ** 2))
        right_clearance = torch.sqrt(torch.sum((normalized_coords[idx_296] - normalized_coords[idx_386]) ** 2))
        brow_eye_clearance = (left_clearance + right_clearance) / 2.0
        
        # 3. Apply spatial jittering (Gaussian noise) if training
        if self.is_train:
            jitter = torch.randn_like(normalized_coords) * 0.01
            normalized_coords = normalized_coords + jitter
            
        # 4. Compute Euclidean distances to 4 nearest anatomical neighbors
        # normalized_coords shape: (88, 2)
        # self.neighbor_tensor shape: (88, 4)
        diff = normalized_coords.unsqueeze(1) - normalized_coords[self.neighbor_tensor] # (88, 4, 2)
        distances = torch.sqrt(torch.sum(diff ** 2, dim=-1)) # (88, 4)
        
        # 5. Concatenate to construct the 6-dimensional feature vector
        feat_6d = torch.cat([normalized_coords, distances], dim=-1) # (88, 6)
        
        # 6. Load image on the fly
        img_tensor = None
        if 'image_paths' in self.data and idx < len(self.data['image_paths']):
            img_path = self.data['image_paths'][idx]
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img)
                except Exception as e:
                    img_tensor = None
        
        if img_tensor is None:
            # Fallback to zero tensor of shape (3, 224, 224)
            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            
        return feat_6d, img_tensor, self.data['labels'][idx]
