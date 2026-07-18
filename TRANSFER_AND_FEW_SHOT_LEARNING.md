# Transfer Learning & Few-Shot Learning in TB Bacilli Detection

This document discusses how **transfer learning** and **few-shot learning** can be
applied to this TB bacilli detection pipeline. It covers what is already in place,
what could be extended, and practical recommendations.

---

## 1. Transfer Learning

### 1.1 Current Usage

Transfer learning is **already used** in this project in the segmentation stage:

| Component | Transfer Learning? | Details |
|---|---|---|
| **SegFormer B4 (Segmentation)** | ✅ Yes | Uses the `nvidia/mit-b4` backbone pretrained on ImageNet-1K. The pretrained encoder weights are loaded and fine-tuned with a differential learning rate (backbone at 0.1× the decoder learning rate). See `tb_config.yaml → model.pretrained: true`. |
| **SwinIR (Super-Resolution)** | ✅ Ready | Infrastructure is in place (`load_pretrained()` method, differential LR, optional encoder freezing). Set `model.pretrained_weights` in `sr_config.yaml` to a pretrained SwinIR checkpoint path to enable. |

### 1.2 Which Pretrained Model Is Used for SwinIR Transfer Learning

The **SwinIR "Classical Image SR" model pretrained on DIV2K + Flickr2K** is the
recommended base model for transfer learning. Specifically:

**Recommended pretrained checkpoint**: `001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`
(the SwinIR-Medium classical SR ×4 checkpoint from the official SwinIR repository)

**Why this specific model?**

Our SwinIR architecture in this project uses the following configuration
(see `configs/sr_config.yaml`):

| Parameter | Our Config | Pretrained SwinIR-M |
|---|---|---|
| `embed_dim` | 180 | 180 ✅ |
| `depths` | [6,6,6,6,6,6,6,6] (**8** blocks) | [6,6,6,6,6,6] (**6** blocks) ⚠️ |
| `num_heads` | [6,6,6,6,6,6,6,6] | [6,6,6,6,6,6] ⚠️ |
| `window_size` | 8 | 8 ✅ |
| `mlp_ratio` | 2.0 | 2.0 ✅ |
| `upscale` | 4 | 4 ✅ |
| `img_channels` | 3 (RGB) | 3 (RGB) ✅ |
| `resi_connection` | "1conv" | "1conv" ✅ |

**Architecture mismatch note**: Our model uses **8 RSTB blocks** while the pretrained
SwinIR-M uses **6 RSTB blocks**. When loaded with `strict=False` (the default in
`load_pretrained()`), the first 6 RSTB blocks plus the shallow feature extractor,
normalization, and reconstruction head will all load pretrained weights successfully.
RSTB blocks 7 and 8 remain randomly initialized and will be trained from scratch.
This is a standard partial transfer learning approach — the majority of the model
(~75% of RSTB blocks) benefits from pretraining.

**Source**: [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
— the official SwinIR repository by the paper authors. The model was pretrained
on the DIV2K (800 images) + Flickr2K (2,650 images) natural image datasets for
general-purpose 4× super-resolution.

**Why SwinIR pretrained on natural images works for microscopy**:

1. **Low-level features transfer**: The shallow convolutional layers and early RSTB
   blocks learn universal low-level features (edges, textures, gradients, contrast
   patterns) that are shared between natural images and microscopy images.
2. **Window-based attention is domain-agnostic**: The Swin Transformer blocks learn
   local spatial relationships within 8×8 windows, which are useful for any image
   domain — including Ziehl-Neelsen stained microscopy slides.
3. **Same task (4× SR)**: Since both the pretrained model and our task perform 4×
   super-resolution, the learned upsampling patterns transfer directly.
4. **Only the high-level features need adaptation**: The deeper RSTB blocks and the
   reconstruction head will adapt to microscopy-specific textures during fine-tuning
   on DDS3, which is why we use a differential learning rate (body at 0.1× the
   head learning rate).

**How to enable** (set in `configs/sr_config.yaml`):

```yaml
model:
  pretrained_weights: "/path/to/pretrained_swinir_x4.pth"

training:
  encoder_lr_mult: 0.1          # Body LR = 2e-5 (10× lower than head)
  freeze_encoder_epochs: 5      # Optional: freeze body for 5 initial epochs
```

**Alternative pretrained models** (also compatible):

| Model | Source | Notes |
|---|---|---|
| SwinIR Classical SR ×4 (DF2K) | Official SwinIR repo | **Recommended** — embed_dim=180 matches, 6/8 RSTB blocks transfer, uses `strict=False` |
| SwinIR Real-World SR ×4 (BSRGAN) | Official SwinIR repo | Same architecture match as above, but includes degradation-aware training — may generalize better to noisy microscopy |
| SwinIR Lightweight SR ×4 | Official SwinIR repo | Smaller model (embed_dim=60) — fewer weights transfer, mostly useful for encoder-only initialization |

### 1.3 Why Transfer Learning Works Here

- **Limited medical data**: The DDS3 dataset has ~5,500 training images across three
  class-balance levels. Pretrained ImageNet features (edges, textures, shapes) provide
  a strong starting point and reduce the risk of overfitting.
- **Domain gap is manageable**: Although microscopy images differ from natural images,
  low-level and mid-level features (edges, blobs, contrast patterns) transfer well.
  The differential learning rate preserves useful pretrained features while letting the
  decoder adapt to the TB-specific task.
- **Faster convergence**: Fine-tuning from pretrained weights typically converges in
  fewer epochs than training from scratch, which is already reflected in the 100-epoch
  segmentation schedule versus the 200-epoch SR schedule.

### 1.4 How Transfer Learning Could Be Extended Further

#### A. Super-Resolution (SwinIR)

SwinIR transfer learning is ready to use (see Section 1.2). The recommended approach:

1. **Download** the pretrained SwinIR Classical SR ×4 checkpoint from the official repo
2. **Set** `model.pretrained_weights` in `sr_config.yaml` to the checkpoint path
3. **Train** with differential LR (`encoder_lr_mult: 0.1`) and optional encoder
   freezing (`freeze_encoder_epochs: 5`)
4. **Expected benefit**: Reduce training from 200 to ~80–120 epochs with equal or
   better PSNR/SSIM, and improve generalization on the DDS3 microscopy domain

Additional options:
- **Pretrained Swin Transformer encoder**: Instead of the full SwinIR checkpoint, load
  only the Swin Transformer encoder weights from an ImageNet-pretrained Swin
  Transformer (e.g., `swin_base_patch4_window7_224`) and adapt the upsampling head.
  This requires `strict=False` since the architecture differs.
- **Benefit**: The microscopy images share common low-level texture and contrast
  patterns with natural images, so pretrained SR features should transfer well and
  reduce the current 200-epoch training requirement.

#### B. Segmentation (SegFormer)

The current setup already uses ImageNet pretraining. Further options include:

- **Domain-specific pretraining**: Fine-tune the backbone on a larger unlabeled
  microscopy image dataset using self-supervised methods (e.g., MAE, DINO, or
  contrastive learning) before fine-tuning on DDS3. This narrows the domain gap
  between ImageNet natural images and microscopy images.
- **Cross-dataset transfer**: If other labeled TB or bacterial microscopy datasets
  are available (e.g., ZNSM-iDB, AutoZion), pretrain or co-train on those first,
  then fine-tune on DDS3. The shared domain makes transfer more effective.
- **Larger backbone variants**: Experiment with `nvidia/mit-b5` or other hierarchical
  Vision Transformer backbones that have been pretrained on larger datasets
  (ImageNet-21K) for richer feature representations.

#### C. End-to-End Pipeline

- **Joint fine-tuning**: After training both stages independently, jointly fine-tune
  the SR → Segmentation pipeline end-to-end so the SR model learns to produce
  outputs optimized for downstream segmentation quality, not just perceptual quality.

---

## 2. Few-Shot Learning

### 2.1 Current Usage

Few-shot learning is **not currently used** in this project. All training follows the
standard supervised paradigm with the full DDS3 training set.

### 2.2 Why Few-Shot Learning Is Relevant

Few-shot learning is highly relevant for medical imaging scenarios where:

- **Annotating microscopy images is expensive**: Creating pixel-level segmentation
  masks for TB bacilli requires expert pathologists and is time-consuming.
- **New TB strains or staining protocols**: When deploying to a new lab or country
  with different staining techniques (e.g., Ziehl-Neelsen vs. Auramine-Rhodamine),
  only a few labeled examples from the new domain may be available.
- **Rare morphologies**: Some bacilli morphologies (beaded, fragmented) may be
  underrepresented in the training set, and few-shot methods could help detect
  them with minimal additional annotation.
- **Rapid deployment**: In resource-limited settings, being able to adapt the model
  with 5–20 labeled images from a new microscope or staining protocol is valuable.

### 2.3 How Few-Shot Learning Could Be Applied

#### A. Metric-Based (Embedding) Approaches

- **Prototypical Networks**: Train the SegFormer encoder to produce embeddings where
  each class (bacilli vs. background) is represented by a prototype (mean embedding
  of support examples). At test time, classify pixels by distance to the nearest
  prototype. This is well-suited to the binary segmentation task.
- **Siamese Networks**: Use a twin-network architecture to compare a query image
  patch with a few labeled support patches, predicting whether each pixel is
  bacilli based on similarity scores.
- **Benefit**: These approaches can adapt to new microscopy domains with as few as
  1–5 labeled examples per class.

#### B. Meta-Learning Approaches

- **MAML (Model-Agnostic Meta-Learning)**: Train the SegFormer model using episodic
  training so that it can rapidly adapt to new tasks (e.g., bacilli detection in
  a new stain or imaging modality) with just a few gradient steps on a small
  support set.
- **Reptile**: A simpler meta-learning alternative to MAML that is easier to
  implement. It trains the model initialization to be easily fine-tunable across
  tasks.
- **Benefit**: Meta-learning provides a model initialization that can adapt quickly
  to new settings, which is ideal for deploying across different clinical sites.

#### C. Few-Shot Segmentation

- **PANet (Prototype Alignment Network)**: A few-shot segmentation method that
  learns to segment new classes given only a few annotated examples. The support
  images provide class prototypes, and the query image is segmented based on
  feature alignment with these prototypes.
- **HSNet (Hypercorrelation Squeeze Network)**: Exploits multi-level feature
  correlations between support and query images for dense prediction, which
  could capture fine-grained bacilli features at multiple scales.
- **Benefit**: These methods are specifically designed for pixel-level prediction
  with limited annotations, matching the segmentation requirement of this project.

#### D. Prompt-Based / Foundation Model Approaches

- **SAM (Segment Anything Model)**: Use Meta's SAM as a foundation model. Given a
  few point or box prompts on bacilli in a new image, SAM can segment the rest
  without any fine-tuning. This effectively provides zero/few-shot segmentation.
- **MedSAM**: A medical-image-specific variant of SAM that may perform better on
  microscopy images out of the box.
- **Benefit**: No training required; the model can be prompted interactively, making
  it practical for rapid deployment in clinical settings.

---

## 3. Practical Recommendations

| Goal | Recommended Approach | Effort |
|---|---|---|
| Improve SR with pretrained weights | Load SwinIR Classical SR ×4 (DIV2K+Flickr2K) checkpoint — **see Section 1.2** | Low |
| Adapt to a new microscope or stain with few labels | Few-shot segmentation (PANet or SAM-based prompting) | Medium |
| Build a single model that generalizes across clinical sites | Meta-learning (MAML/Reptile on SegFormer) | High |
| Reduce annotation cost for new datasets | SAM/MedSAM with interactive prompting | Low–Medium |
| Maximize segmentation accuracy on DDS3 | Self-supervised pretraining on unlabeled microscopy data + fine-tune | Medium–High |

### Summary

- **Transfer learning** is a core part of this project:
  - **SegFormer**: Uses ImageNet-pretrained `nvidia/mit-b4` backbone (already active)
  - **SwinIR**: Uses pretrained SwinIR Classical SR ×4 from DIV2K+Flickr2K
    (infrastructure ready — set `model.pretrained_weights` in `sr_config.yaml`)
- **The pretrained model for SwinIR** is the official SwinIR-M ×4 checkpoint
  (embed_dim=180, 6 RSTB blocks, window_size=8). Our model extends this to 8 RSTB
  blocks — the first 6 blocks load pretrained weights via `strict=False`, blocks 7–8
  train from scratch. See Section 1.2 for full details and architecture comparison.
- **Few-shot learning** is not yet used but is highly applicable for scenarios
  involving limited labeled data, new imaging domains, or rapid deployment across
  clinical sites. Metric-based and prompt-based approaches offer the most practical
  paths to adoption with the existing architecture.
