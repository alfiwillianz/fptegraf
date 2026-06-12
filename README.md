# DeepFacialGAT: 88-Landmark Graph Attention Network for Emotion Recognition

DeepFacialGAT is an advanced, high-performance facial expression recognition engine. The system integrates geometric graph topology, multi-head graph attention mechanisms, and dense local texture features into a unified deep learning model running entirely client-side on the edge via ONNX Runtime Web.

---

## 📐 Graph Construction & Feature Engineering

Instead of treating landmarks as simple, flat coordinate coordinates, DeepFacialGAT constructs an **Anatomical landmark Graph** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ consisting of **88 selected landmark nodes** optimized for facial expression deformation tracking.

```
                  [ 88 Facial Landmarks ]
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    Scale & Translation          Anatomical Neighbors
     (Interocular Dist)            (Floyd-Warshall)
              │                           │
              └─────────────┬─────────────┘
                            ▼
              [ 6-Dimensional Node Vector ]
               [ x, y, d1, d2, d3, d4 ]
```

### 1. Coordinate Normalization
To achieve scale, translation, and rotation invariance, raw landmarks undergo spatial transformation:
* **Translation Invariance:** Nodes are centered relative to the Nose Tip anchor ($idx=1$):
  $$\vec{x}'_i = \vec{x}_i - \vec{x}_{\text{nose}}$$
* **Scale Invariance:** Centered coordinates are divided by the **Interocular Distance** ($d_{\text{eye}}$), computed dynamically as the Euclidean distance between the medial corners of the left and right eyes:
  $$\vec{x}''_i = \frac{\vec{x}'_i}{d_{\text{eye}}}$$

### 2. Topological Neighbor Connectivity
We precompute a static anatomical adjacency matrix mapping each of the 88 landmarks to its 4 closest anatomical neighbors using the Floyd-Warshall shortest path algorithm. 

### 3. Node Feature Representation (6-Dimensional Vectors)
For each node $i \in \mathcal{V}$, we construct a 6-dimensional feature vector:
$$\vec{f}_i = [x''_i, y''_i, d_{i,1}, d_{i,2}, d_{i,3}, d_{i,4}]$$
where:
* $(x''_i, y''_i)$ is the scale-invariant 2D coordinate.
* $d_{i,k}$ is the scale-invariant Euclidean distance to its $k$-th anatomical neighbor.

---

## 🧠 Model Architecture

The model uses a dual-stream stream fusion design where spatial coordinate graphs and localized facial textures are processed in parallel, then fused late in the network.

```
       [ Input Coordinates ]                 [ Input Face Canvas ]
         (1, 88, 6)                            (1, 3, 224, 224)
             │                                        │
             ▼                                        ▼
      [ GAT Backbone ]                         [ Texture Stream ]
    (4 GAT layers, 4 heads)                   (Feature Extractor)
             │                                        │
             ▼ (Global Average Pooling)               ▼ (Adaptive Pooling)
      [ 256-D GAT Vector ]                    [ 960-D Texture Vector ]
             │                                        │
             └───────────────────┬────────────────────┘
                                 ▼ (Concatenation)
                       [ 1216-D Fused Vector ]
                                 │
                                 ▼ (MLP Classifier)
                        [ Emotion Probabilities ]
```

### 1. Graph Attention Stream
The coordinate features pass through a 4-layer Graph Attention Network.
* **Input Projection:** Maps the 6D node features into a 256-dimensional hidden representation.
* **High-Capacity Attention Layer:** Computes attention weights across the graph using 4 independent attention heads:
  $$\alpha_{ij}^{(h)} = \frac{\exp\left(\text{LeakyReLU}\left(\vec{a}^{T}_h [W_h \vec{h}_i \,\|\, W_h \vec{h}_j] + B_{ij}^{(h)}\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\vec{a}^{T}_h [W_h \vec{h}_i \,\|\, W_h \vec{h}_k] + B_{ik}^{(h)}\right)\right)}$$
  where:
  * $W_h$ is the linear projection weight matrix for head $h$.
  * $\vec{a}_h$ is the learnable attention coefficient vector.
  * $B_{ij}^{(h)}$ is a learnable spatial bias representing anatomical distance constraints.
  * $\mathcal{N}_i$ is the neighborhood boundary of node $i$.
* **Global Pooling:** An average pooling layer aggregates node embeddings across all 88 landmarks to output a 256-dimensional feature vector.

### 2. Dense Texture Stream
To complement geometric landmarks (which miss subtle micro-expressions like forehead wrinkles, cheek bloating, or lip thinness), the model processes a normalized face canvas via a dense convolutional feature extractor.
* The face area is cropped dynamically in the browser with 20% padding, resized to $224 \times 224$, and normalized using ImageNet parameters.
* The convolutional stream extracts local texture maps, pooling them to a 960-dimensional representation.

### 3. Feature Late-Fusion
The 256-dimensional graph embedding and the 960-dimensional texture representation are concatenated to produce a unified 1216-dimensional vector. This vector is passed to a classification head (Fully Connected -> ReLU -> Dropout -> Linear) that outputs probabilities for 7 emotion categories:
* *Surprise*
* *Fear*
* *Disgust*
* *Happiness*
* *Sadness*
* *Anger*
* *Neutral*

---

## 🚀 Pipeline & Execution Modes

### 1. Client-Side Edge Inference (Browser)
In edge mode, inference runs entirely locally in the browser using ONNX Runtime Web.
1. MediaPipe FaceMesh extracts 468 landmark coordinates.
2. The UI extracts the 88 selected landmark indices and normalizes coordinates.
3. The UI crops the face canvas with 20% padding and applies ImageNet normalization.
4. Both inputs are fed to the `gcn_emotion.onnx` session:
   ```javascript
   const output = await onnxSession.run({ 
       input_coords: coordsTensor, // Shape: [1, 88, 6]
       input_img: imgTensor        // Shape: [1, 3, 224, 224]
   });
   ```

### 2. Server-Side WebSocket Stream
The model can also run on a GPU-enabled backend using the FastAPI server.
1. The frontend streams landmarks and optional base64 image strings over WebSocket to `/stream`.
2. The server processes the 6D graph features and normalizes the image payload.
3. Running inference:
   ```python
   logits, attention = model(feat_6d, img_tensor)
   ```
4. Returns the predicted emotion probabilities and GAT attention overlays to draw glowing connection lines.

---

## 🛠️ Usage Instructions

### Training the Model
To train the model on your preprocessed datasets:
```bash
python -m train --epochs 350 --batch-size 128 --lr 1e-3
```
* **Split LR Optimizer:** Pretrained convolutional features are optimized at `1e-4` to prevent vanishing gradients, while GAT and fusion heads use `1e-3`.

### Exporting to ONNX
To export the trained PyTorch state dict to ONNX format:
```bash
python export_compat.py
```
This produces `gcn_emotion.onnx` ready to be served to `index.html`.

Here is a complete, production-ready Typst template engineered strictly to the competition rules provided in `image_85c51e.jpg`, `image_85c521.jpg`, `image_85c525.jpg`, and `image_85c53d.jpg`.

It implements a modular layout setup: **Cover page** (no numbering), **Preliminary pages** (lowercase Roman numerals, bottom-right), and **Main Content** (Arabic numerals starting back at 1, bottom-right), with the exact mandated margins, line spacing, and systematic structure.

---

## 1. The Core Template Function (`template.typ`)

Save this code block as `template.typ`. This defines the entire document logic, margins, pagination switches, and typography constraints.

```typst
#let technical_report(
  title: "",
  team_name: "",
  institution: "",
  category: "",
  year: "",
  logo: none,
  body
) = {
  // 1. Global Document Setup (A4, Times New Roman, Size 12)
  set page(
    paper: "a4",
    margin: (top: 3cm, bottom: 3cm, left: 4cm, right: 3cm),
  )
  set text(font: "Times New Roman", size: 12pt, lang: "en")
  set par(justify: true, leading: 1.15em) // Compliant with 1.15 - 1.5 line spacing

  // Heading Styling
  show heading: set text(weight: "bold")
  show heading: set par(leading: 1.15em)
  
  // Style Headings to match academic report requirements
  show heading.where(level: 1): it => block(width: 100%, below: 1.5em, above: 2em)[
    #set text(size: 14pt)
    #it.body
  ]
  show heading.where(level: 2): it => block(below: 1em, above: 1.5em)[
    #set text(size: 12pt)
    #it.body
  ]

  // --- 2. COVER PAGE (No Page Numbering) ---
  page(header: none, footer: none)[
    #set align(center)
    #v(1cm)
    
    // Title Block
    #block(width: 80%)[
      #set text(size: 16pt, weight: "bold")
      #title
    ]
    
    #v(3cm)
    
    // Institution Logo Placeholder
    #if logo != none {
      box(width: 4cm, height: 4cm, logo)
    } else {
      rect(width: 4cm, height: 4cm, stroke: 1pt + gray, radius: 4pt)[
        #set align(center + horizon)
        #text(size: 10pt, fill: gray, [INSTITUTION\ LOGO])
      ]
    }
    
    #v(3cm)
    
    // Team Metadata
    #text(size: 12pt)[
      *Team Name:* #team_name \
      *Category:* #category \
      *Institution:* #institution \
    ]
    
    #v(keep: true, 1fr)
    
    // Submission Year
    #text(size: 12pt, weight: "bold")[
      #year
    ]
    #v(1cm)
  ]

  // --- 3. PRELIMINARY PAGES (Roman Numerals: i, ii, iii... Bottom-Right) ---
  set page(
    footer: locate(loc => {
      let page_num = counter(page).at(loc).first()
      align(right, text(size: 10pt, numbering("i", page_num)))
    })
  )
  
  // Abstract / Executive Summary (Often expected in preliminaries before ToC)
  heading(level: 1, numbering: none)[Abstract]
  [
    #lorem(120)
  ]
  #pagebreak()

  // Table of Contents
  heading(level: 1, numbering: none)[Table of Contents]
  outline(title: none, depth: 3, indent: 1.5em)
  #pagebreak()

  // --- 4. MAIN CONTENT PAGES (Arabic Numerals: 1, 2, 3... Bottom-Right) ---
  // Reset page counter back to 1 for the main content
  counter(page).update(1)
  set page(
    footer: locate(loc => {
      let page_num = counter(page).at(loc).first()
      align(right, text(size: 10pt, numbering("1", page_num)))
    })
  )

  body
}

```

---

## 2. The Main Document Content (`main.typ`)

Save this code block as `main.typ` in the same directory. It imports your template and structural layout exactly matching Chapters I through V plus appendices.

```typst
