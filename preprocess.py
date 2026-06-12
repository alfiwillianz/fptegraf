import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import mediapipe as mp

def main():
    # 1. Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False, # Standard 468 landmarks
        min_detection_confidence=0.5
    )

    os.makedirs("data", exist_ok=True)

    # 2. Define Unified Class Mappings
    raf_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    affectnet_mapping = {
        "surprise": 0, "fear": 1, "disgust": 2, 
        "happy": 3, "sad": 4, "anger": 5, "neutral": 6
    }

    # 3. Scan AffectNet directories and perform Train/Test Split
    print("=== Scanning AffectNet directories ===")
    affectnet_dir = os.path.join("DATASET", "affectnet")
    affectnet_paths = []
    affectnet_labels = []
    
    if os.path.exists(affectnet_dir):
        for folder_name, mapped_label in affectnet_mapping.items():
            folder_path = os.path.join(affectnet_dir, folder_name)
            if os.path.isdir(folder_path):
                # Only list image files
                img_names = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_name in img_names:
                    affectnet_paths.append(os.path.join(folder_path, img_name))
                    affectnet_labels.append(mapped_label)
        
        print(f"Total AffectNet images found: {len(affectnet_paths)}")
        
        # Split 90% train / 10% test
        an_train_paths, an_test_paths, an_train_labels, an_test_labels = train_test_split(
            affectnet_paths, affectnet_labels, test_size=0.10, random_state=42, stratify=affectnet_labels
        )
        print(f"AffectNet split: {len(an_train_paths)} train, {len(an_test_paths)} test")
    else:
        print(f"[Error] AffectNet directory not found at {affectnet_dir}!")
        an_train_paths, an_test_paths, an_train_labels, an_test_labels = [], [], [], []

    an_splits = {
        "train": (an_train_paths, an_train_labels),
        "test": (an_test_paths, an_test_labels)
    }

    splits = ["train", "test"]

    for split in splits:
        print(f"\n==========================================")
        print(f"Processing Combined Data for Split: {split.upper()}")
        print(f"==========================================")
        
        coords_list = []
        labels_list = []
        image_paths_list = []
        
        # --- PHASE 1: PROCESS RAF-DB IMAGES ---
        csv_filename = os.path.join("DATASET", "raf-db", f"{split}_labels.csv")
        if os.path.exists(csv_filename):
            print(f"Ingesting RAF-DB via {csv_filename}...")
            df = pd.read_csv(csv_filename)
            raf_success, raf_skipped = 0, 0
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="RAF-DB"):
                img_name = row['image']
                raw_label = int(row['label'])
                
                if raw_label not in raf_mapping:
                    continue
                mapped_label = raf_mapping[raw_label]
                
                # Image path is DATASET/raf-db/<split>/<label>/<img_name>
                img_path = os.path.join("DATASET", "raf-db", split, str(raw_label), img_name)
                
                if not os.path.exists(img_path):
                    raf_skipped += 1
                    continue
 
                img = cv2.imread(img_path)
                if img is None:
                    raf_skipped += 1
                    continue
 
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(img_rgb)
                
                if not results.multi_face_landmarks:
                    raf_skipped += 1
                    continue
                
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [[lm.x, lm.y] for lm in face_landmarks.landmark[:468]]
                
                coords_list.append(torch.tensor(landmarks, dtype=torch.float32))
                labels_list.append(mapped_label)
                image_paths_list.append(img_path)
                raf_success += 1
                
            print(f"-> RAF-DB Ingest: {raf_success} processed, {raf_skipped} skipped/failed.")
        else:
            print(f"[Warning] {csv_filename} not found. Skipping RAF-DB pipeline block.")
 
        # --- PHASE 2: PROCESS AFFECTNET IMAGES ---
        an_paths, an_labels = an_splits[split]
        if len(an_paths) > 0:
            print(f"Ingesting AffectNet ({split} partition)...")
            an_success, an_skipped = 0, 0
            for img_path, mapped_label in tqdm(zip(an_paths, an_labels), total=len(an_paths), desc="AffectNet"):
                if not os.path.exists(img_path):
                    an_skipped += 1
                    continue
 
                img = cv2.imread(img_path)
                if img is None:
                    an_skipped += 1
                    continue
 
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(img_rgb)
                
                if not results.multi_face_landmarks:
                    an_skipped += 1
                    continue
                
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [[lm.x, lm.y] for lm in face_landmarks.landmark[:468]]
                
                coords_list.append(torch.tensor(landmarks, dtype=torch.float32))
                labels_list.append(mapped_label)
                image_paths_list.append(img_path)
                an_success += 1
            print(f"-> AffectNet Ingest: {an_success} processed, {an_skipped} skipped/failed.")
        else:
            print(f"No AffectNet images to process for {split} split.")
 
        # --- PHASE 3: CONSOLIDATE & EXPORT ---
        if len(coords_list) > 0:
            coords_tensor = torch.stack(coords_list)   # (Total_Samples, 468, 2)
            labels_tensor = torch.tensor(labels_list, dtype=torch.long) # (Total_Samples,)
            
            output_path = os.path.join("data", f"{split}_data.pt")
            torch.save({
                "coords": coords_tensor,
                "labels": labels_tensor,
                "image_paths": image_paths_list
            }, output_path)
            
            print(f"🎉 Successfully exported Combined {split.upper()} Data!")
            print(f"Total Tensors Shape: {coords_tensor.shape} | File Saved to: {output_path}")
        else:
            print(f"[Error] Zero valid landmarks extracted for split: {split}")

    face_mesh.close()
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
