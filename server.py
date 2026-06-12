import os
import json
import base64
import io
import torch
import uvicorn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from train.model import DeepFacialGAT

app = FastAPI(title="DeepFacialGAT Inference Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Server backend mounting active on: {device}")

NEIGHBORS = [
    [11, 51, 12, 52], [2, 3, 4, 5], [1, 3, 4, 5], [4, 2, 1, 5], [3, 5, 2, 42], [42, 4, 3, 2], [48, 41, 10, 30], [21, 61, 20, 60], [23, 63, 44, 83], [22, 62, 45, 84], [48, 40, 6, 39], [0, 12, 13, 51], [13, 11, 46, 0], [46, 12, 14, 11], [46, 32, 13, 25], [17, 27, 16, 28], [27, 28, 15, 17], [15, 28, 27, 16], [47, 26, 24, 19], [47, 20, 18, 21], [19, 21, 47, 7], [20, 7, 19, 61], [45, 9, 25, 62], [44, 8, 24, 63], [26, 44, 18, 23], [32, 45, 14, 22], [24, 18, 47, 44], [15, 16, 17, 28], [16, 17, 27, 15], [35, 43, 34, 36], [41, 31, 6, 33], [33, 30, 41, 34], [14, 25, 46, 45], [34, 31, 35, 30], [35, 33, 29, 31], [29, 34, 43, 33], [43, 37, 29, 38], [38, 36, 43, 39], [37, 39, 40, 36], [40, 38, 10, 37], [39, 10, 48, 38], [6, 30, 48, 31], [5, 4, 3, 2], [29, 36, 35, 37], [24, 23, 26, 8], [25, 22, 32, 9], [14, 13, 32, 12], [18, 19, 26, 20], [10, 6, 40, 41], [87, 81, 50, 70], [87, 80, 49, 79], [0, 52, 53, 11], [53, 51, 85, 0], [85, 52, 54, 51], [85, 72, 53, 65], [57, 67, 56, 68], [67, 68, 55, 57], [55, 68, 67, 56], [86, 66, 64, 59], [86, 60, 58, 61], [59, 61, 86, 7], [60, 7, 59, 21], [84, 9, 65, 22], [83, 8, 64, 23], [66, 83, 58, 63], [72, 84, 54, 62], [64, 58, 86, 83], [55, 56, 57, 68], [56, 57, 67, 55], [75, 82, 74, 76], [81, 71, 49, 73], [73, 70, 81, 74], [54, 65, 85, 84], [74, 71, 75, 70], [75, 73, 69, 71], [69, 74, 82, 73], [82, 77, 69, 78], [78, 76, 82, 79], [77, 79, 80, 76], [80, 78, 50, 77], [79, 50, 87, 78], [49, 70, 87, 71], [69, 76, 75, 77], [64, 63, 66, 8], [65, 62, 72, 9], [54, 53, 72, 52], [58, 59, 66, 60], [50, 49, 80, 81]
]

# Initialize model instance
model = DeepFacialGAT(num_nodes=88, hidden_dim=256, depth=4).to(device)

# Look for standard checkpoints
checkpoint_path = "best_gat_f1_model.pth"
if not os.path.exists(checkpoint_path) and os.path.exists("model.pth"):
    checkpoint_path = "model.pth"
    print("Using fallback checkpoint: model.pth")

if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
        print(f"Loaded GAT checkpoint weights successfully from '{checkpoint_path}'.")
    except Exception as e:
        print(f"Direct load failed: {e}. Trying load_state_dict with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint weights with strict=False from '{checkpoint_path}'.")
else:
    print(f"[Warning] Checkpoint missing! Initializing fallback random weights. Expected model at '{checkpoint_path}'.")
model.eval()

# Common image transforms for model input
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        neighbor_tensor = torch.tensor(NEIGHBORS, dtype=torch.long, device=device)
        while True:
            message = await websocket.receive_text()
            raw_data = json.loads(message)
            
            # Handle both single-modality coordinates (flat list) and dual-modality (dict) payloads
            if isinstance(raw_data, dict) and "coords" in raw_data:
                coords_list = raw_data["coords"]
            else:
                coords_list = raw_data

            n_floats = len(coords_list)
            n_nodes = n_floats // 2
            
            # Reshape coordinates to [1, 88, 2]
            coords_tensor = torch.tensor(coords_list, dtype=torch.float32).view(1, n_nodes, 2).to(device)
            if coords_tensor.size(1) > 88:
                coords_tensor = coords_tensor[:, :88, :]
                
            # Perform 6D feature extraction: (1, 88, 6)
            diff = coords_tensor.unsqueeze(2) - coords_tensor[:, neighbor_tensor] # (1, 88, 4, 2)
            distances = torch.sqrt(torch.sum(diff ** 2, dim=-1)) # (1, 88, 4)
            feat_6d = torch.cat([coords_tensor, distances], dim=-1) # (1, 88, 6)
            
            # Extract image tensor if present in websocket payload
            img_tensor = None
            if isinstance(raw_data, dict) and "image" in raw_data:
                try:
                    img_b64 = raw_data["image"]
                    if img_b64.startswith("data:"):
                        img_b64 = img_b64.split(",")[1]
                    img_bytes = base64.b64decode(img_b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img_tensor = image_transform(img).unsqueeze(0).to(device)
                except Exception as e:
                    print(f"Failed to decode incoming image payload: {e}")
                    img_tensor = None
                    
            if img_tensor is None:
                img_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)
                
            with torch.no_grad():
                logits, attention_matrix = model(feat_6d, img_tensor)
                probabilities = F.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
                predicted_class = torch.argmax(logits, dim=1).item()
                
            attn_list = attention_matrix[0].cpu().numpy().tolist()
            
            payload = {
                "emotion_id": predicted_class,
                "probabilities": probabilities,
                "attention_matrix": attn_list
            }
            await websocket.send_text(json.dumps(payload))
            
    except WebSocketDisconnect:
        print("Stream connection disconnected cleanly.")
    except Exception as e:
        print(f"Active runtime fault encountered: {e}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
