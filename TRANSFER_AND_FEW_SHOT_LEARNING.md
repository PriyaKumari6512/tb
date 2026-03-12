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
| **SwinIR (Super-Resolution)** | ❌ No | Trained from scratch on the DDS3 dataset with staged fine-tuning (L1 loss → L1 + perceptual loss). |

### 1.2 Why Transfer Learning Works Here

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

### 1.3 How Transfer Learning Could Be Extended Further

#### A. Super-Resolution (SwinIR)

SwinIR is currently trained from scratch. Transfer learning could be applied by:

- **Pretrained SwinIR weights**: Load weights from a SwinIR model pretrained on
  general-purpose SR benchmarks (DIV2K, Flickr2K). This would provide a strong
  initialization for the Swin Transformer blocks and reduce training time.
- **Pretrained Swin Transformer encoder**: Instead of the full SwinIR checkpoint, load
  only the Swin Transformer encoder weights from an ImageNet-pretrained Swin
  Transformer (e.g., `swin_base_patch4_window7_224`) and adapt the upsampling head.
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
| Improve current pipeline with available data | Extend transfer learning to SwinIR (load pretrained SR weights) | Low |
| Adapt to a new microscope or stain with few labels | Few-shot segmentation (PANet or SAM-based prompting) | Medium |
| Build a single model that generalizes across clinical sites | Meta-learning (MAML/Reptile on SegFormer) | High |
| Reduce annotation cost for new datasets | SAM/MedSAM with interactive prompting | Low–Medium |
| Maximize segmentation accuracy on DDS3 | Self-supervised pretraining on unlabeled microscopy data + fine-tune | Medium–High |

### Summary

- **Transfer learning** is already a core part of this project (SegFormer with
  ImageNet pretraining) and can be extended to SwinIR and through domain-specific
  pretraining to further improve performance.
- **Few-shot learning** is not yet used but is highly applicable for scenarios
  involving limited labeled data, new imaging domains, or rapid deployment across
  clinical sites. Metric-based and prompt-based approaches offer the most practical
  paths to adoption with the existing architecture.
