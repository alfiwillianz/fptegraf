import os
import torch
import onnx
from train.model import DeepFacialGAT

def main():
    device = torch.device("cpu")
    print("Initializing model...")
    model = DeepFacialGAT(num_nodes=88, hidden_dim=256, num_classes=7, depth=4).to(device)
    
    # Load model weights if available
    checkpoint_path = "model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint)
            print("Loaded GAT checkpoint weights successfully.")
        except Exception as e:
            print(f"Direct load failed: {e}. Attempting strict=False...")
            model.load_state_dict(checkpoint, strict=False)
            print("Loaded checkpoint weights with strict=False.")
    else:
        print("[Warning] Checkpoint model.pth not found. Exporting random weights.")
    
    model.eval()

    # Create dummy inputs: coords (shape [1, 88, 6]) and image (shape [1, 3, 224, 224])
    dummy_coords = torch.zeros((1, 88, 6), dtype=torch.float32)
    dummy_img = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    print("Exporting model to ONNX using modern opset_version=18...")
    torch.onnx.export(
        model,
        (dummy_coords, dummy_img),
        "gcn_emotion.onnx",
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_coords', 'input_img'],
        output_names=['emotion_probabilities', 'attention_matrix'],
        dynamo=False,  # Force legacy TorchScript exporter
        dynamic_axes={
            'input_coords': {0: 'batch_size'},
            'input_img': {0: 'batch_size'},
            'emotion_probabilities': {0: 'batch_size'},
            'attention_matrix': {0: 'batch_size'}
        }
    )
    print("Export completed successfully!")

    # Flatten the model to be self-contained (inline weights)
    print("Loading and saving with onnx library to merge external data...")
    onnx_model = onnx.load("gcn_emotion.onnx")
    
    # Remove external data references and inline everything
    onnx.save(onnx_model, "gcn_emotion.onnx")
    print("Model flattened successfully! Saved to gcn_emotion.onnx")

    # Verify size
    size_kb = os.path.getsize("gcn_emotion.onnx") / 1024
    print(f"Final file size: {size_kb:.2f} KB")

if __name__ == "__main__":
    main()
