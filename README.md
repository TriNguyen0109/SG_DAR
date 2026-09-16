# SG-DAR: Sketch-Guided Diffusion Augmented Retrieval

Source code implementation and evaluation for **Multimodal Conversational Image Retrieval** combining **Dialogue Text** and **Freehand Sketches** using **Stable Diffusion 3 + Softedge ControlNet**.

---

## 📁 Directory Structure

```
baseline/main/
├── HUONG_DAN_CHAY.md                  # 📘 Execution guide documentation (Vietnamese)
├── sd3_text_sketch.py                 # SG-DAR Image generation (SD3 + ControlNet Softedge)
├── sd3_text.py                        # Original DAR Image generation (SD3 Text-only)
├── eval_feature_fusion.py             # Feature Fusion evaluation (Fine-tuned ChatIR BLIP)
├── eval_feature_fusion_zeroshot.py    # Feature Fusion evaluation (Zero-Shot BLIP)
├── baselines.py                       # BLIP & CLIP feature extraction utilities
├── run.sh                             # Helper script for running in Docker
├── run_docker.sh                      # Docker build and run script
├── VisDial_v1_0_queries_val_sketch.json  # 1,000 validation dialogues with sketch paths
├── Search_Space_val_50k.json          # Search space consisting of 50,000 candidate images
├── generated_images_text_sketch/      # Generated images directory for SG-DAR (B4)
└── generated_images_text/             # Generated images directory for original DAR (B3)
```

---

## ⚡ Quick Start Commands

For detailed parameters, refer to **[HUONG_DAN_CHAY.md](file:///workingspace_aiclub/WorkingSpace/Personal/core_baotg/tri/baseline/main/HUONG_DAN_CHAY.md)**.

### 1. Generate SG-DAR Images (Text + Sketch):
```bash
python sd3_text_sketch.py --offload none --skip_existing
```

### 2. Generate DAR Images (Text-only):
```bash
python sd3_text.py --offload none --skip_existing
```

### 3. Evaluate Benchmark (Hits@10):
```bash
# Evaluate with VisDial Fine-tuned BLIP (ChatIR weights)
python eval_feature_fusion.py --model_type chatir --config all --batch_size 128

# Evaluate with COCO Retrieval Fine-tuned BLIP
python eval_feature_fusion.py --model_type cocoft --config all --batch_size 128

# Evaluate with Pure Zero-Shot BLIP (129M Base)
python eval_feature_fusion.py --model_type zs --config all --batch_size 128

# Evaluate all 3 models sequentially
python eval_feature_fusion.py --model_type all --config all --batch_size 128
```

---

## 📊 Experimental Baselines Table

| Configuration | Baseline Name | Input Modalities | Query Formulation (Feature Fusion) |
| :---: | :--- | :--- | :--- |
| **`B1`** | **Text-only** | Dialogue Text | $v_{\text{query}} = v_{\text{text}}$ |
| **`B2`** | **Direct Sketch** | Dialogue Text + Freehand Sketch | $v_{\text{query}} = \text{Norm}(w_a v_{\text{text}} + w_g v_{\text{sketch}})$ |
| **`B3`** | **DAR** *(Original)* | Dialogue Text + Text-to-Image | $v_{\text{query}} = \text{Norm}(w_a v_{\text{text}} + w_b v_{\text{gen}})$ |
| **`B4`** | **SG-DAR** *(Proposed)* | Dialogue Text + Text-to-Image + Sketch | $v_{\text{query}} = \text{Norm}(w_a v_{\text{text}} + w_b v_{\text{gen}} + w_g v_{\text{sketch}})$ |