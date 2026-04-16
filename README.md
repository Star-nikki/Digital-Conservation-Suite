# 🎨 Digital Conservation Suite

> A Dockerized, multi-algorithm pipeline for art authentication, brushstroke analysis,
> pigment extraction, and style classification — powered by The Met Museum Open Access API.

---

## Architecture

```
Met Museum API  ──►  stream_image()  ──►  ┌─────────────────────────────┐
                                          │  4-Algorithm Pipeline        │
                                          │                              │
                                          │  1. YOLOv8   (composition)   │
                                          │  2. GLCM     (texture)       │
                                          │  3. K-Means  (palette)       │
                                          │  4. ResNet50 (style/era)     │
                                          └────────────┬────────────────┘
                                                       │
                                                       ▼
                                          Streamlit Conservation Dashboard
```

---

## Project Structure

```
art-conservation-suite/
├── algorithms.py        # Core CV pipeline (4 algorithms)
├── app.py               # Streamlit dashboard
├── requirements.txt     # Python dependencies
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # One-command local spin-up
└── README.md
```

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
git clone <repo>
cd art-conservation-suite

# Build and start (first build downloads ~2 GB of model weights)
docker compose up --build

# Open in browser
open http://localhost:8501
```

### Option B — Local Python

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

---

## Usage

1. Open **http://localhost:8501**
2. Enter a Met Museum Object ID in the sidebar (or click a sample)
3. Click **▶ Analyse Artwork**
4. The pipeline streams the image, runs all 4 algorithms, and displays a full
   Conservation Dashboard with a downloadable report.

### Sample Object IDs

| Object ID | Artwork                             |
|-----------|-------------------------------------|
| `436535`  | Van Gogh — Wheat Field with Cypresses |
| `459055`  | Monet — La Grenouillère             |
| `670906`  | Vermeer — Young Woman with a Water Pitcher |
| `10927`   | Goya — Don Manuel Osorio Manrique   |

---

## The 4-Algorithm Pipeline

### 1. YOLOv8 — Compositional Layout Detection (`detect_composition`)

Uses the **YOLOv8 nano** object detector to identify figures, objects, and
regions of interest. Overlays a **rule-of-thirds grid** and scores how well
the artist's key elements align with golden-ratio intersections.

- **Output:** Bounding boxes, composition type label, rule-of-thirds score
- **Library:** `ultralytics`

### 2. GLCM — Brushstroke Texture Analysis (`analyze_texture`)

Computes a **Gray-Level Co-occurrence Matrix** at three distances and four
angles, extracting contrast, energy, homogeneity, correlation, dissimilarity,
and **Shannon entropy**. Thresholds map to artistic movements:

| Entropy | Contrast | Inferred Style          |
|---------|----------|-------------------------|
| > 4.5   | > 150    | Impressionist           |
| > 3.5   | > 80     | Baroque / Romantic      |
| < 3.0   | < 40     | Renaissance / Neoclassical |
| > 5.0   | any      | Abstract Expressionist  |

- **Library:** `scikit-image`

### 3. K-Means — Pigment & Palette Extraction (`extract_palette`)

Clusters the pixel colour space into **5 dominant pigments** using K-Means
(k=5, n_init=10). Each cluster centre is mapped to a historical pigment name
(e.g., *Lapis Lazuli*, *Vermillion*, *Yellow Ochre*) via HSV thresholds.

- **Output:** Hex codes, RGB values, coverage percentages, pigment IDs, swatch
- **Library:** `scikit-learn`

### 4. ResNet-50 — Style & Era Classification (`classify_style`)

A **pre-trained ResNet-50** (ImageNet-1K V2) acts as a broad feature extractor.
Top-5 activations are mapped to artistic eras via a keyword dictionary; when
no keyword matches, heuristics on brightness and saturation select the era.

- **Output:** Era label, era description, confidence score, top-5 features
- **Library:** `torch`, `torchvision`

---

## Docker Details

The Dockerfile uses a **multi-stage build**:

| Stage   | Purpose                                      | Included in final image? |
|---------|----------------------------------------------|--------------------------|
| builder | Compiles wheels, downloads model weights     | ✗ (discarded)            |
| runtime | Copies venv + weights + app code only        | ✓                        |

Model weights are pre-cached during the build so the container works **offline**
after the initial pull.

---

## API Reference — The Met

The pipeline calls two Met endpoints:

```
GET https://collectionapi.metmuseum.org/public/collection/v1/objects/{objectID}
```

Returns metadata including `primaryImage` — the full-resolution JPEG URL that
is **streamed directly** into the pipeline without disk I/O.

---

## Environment Variables

| Variable                            | Default | Description                      |
|-------------------------------------|---------|----------------------------------|
| `STREAMLIT_SERVER_PORT`             | `8501`  | Port to expose                   |
| `STREAMLIT_SERVER_HEADLESS`         | `true`  | Disable browser auto-open        |
| `STREAMLIT_SERVER_ENABLE_CORS`      | `false` | CORS for embedded deployments    |

---

## Extending the Pipeline

- **Add an algorithm:** Implement a new function in `algorithms.py` and call it
  in the `# Run the 4 algorithms` section of `app.py`.
- **Change YOLO model:** Replace `yolov8n.pt` with `yolov8s.pt` or `yolov8m.pt`
  in `algorithms.py` for higher accuracy at the cost of speed.
- **GPU support:** Install `torch` with CUDA and the container will automatically
  use GPU if one is available (`device = "cuda" if torch.cuda.is_available()`).

---

## License

The Met Museum's images are made available under the
[Creative Commons Zero (CC0)](https://creativecommons.org/publicdomain/zero/1.0/)
licence. Application code is MIT.
