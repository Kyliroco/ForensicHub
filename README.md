<p align="center" width="100%"> 
<img src="images/logo.jpg" alt="OSQ" style="width: 28%; min-width: 150px; display: block; margin: auto;">
</p>

# [NeurlPS 2025] ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization

<div align="center">

[Bo Du](https://github.com/dddb11)†,    [Xuekang Zhu](https://github.com/Inkyl)†, [Xiaochen Ma](https://ma.xiaochen.world/)†,    [Chenfan Qu](https://github.com/qcf-568)†, Kaiwen Feng†, Zhe Yang  
[Chi-Man Pun](https://cmpun.github.io/),    Jian Liu*, [Jizhe Zhou](https://knightzjz.github.io/)*   

<div align="center"><span style="font-size: smaller;">
<br>†: joint first author & equal contribution
*: corresponding author</br>  
</div>  
</div>


******
[![Arxiv](https://img.shields.io/badge/arXiv-2505.11003-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2505.11003)
[![Documents](https://img.shields.io/badge/Documents-Click_here-brightgreen?logo=read-the-docs)](https://scu-zjz.github.io/ForensicHub-doc/)
![license](https://img.shields.io/github/license/scu-zjz/ForensicHub?logo=license)
<!----
[![Ask Me Anything!](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://GitHub.com/scu-zjz/) 
---->
🙋‍♂️Welcome to **ForensicHub**!   

ForensicHub is the go-to benchmark and modular codebase for all-domain fake image detection and localization,
covering deepfake detection (Deepfake), image manipulation detection and localization (IMDL), artificial
intelligence-generated image detection (AIGC), and document image manipulation localization (Doc). Whether you're
benchmarking forensic models or building your own cross-domain pipelines, **ForensicHub** offers a flexible, configuration-driven
architecture to streamline development, comparison, and analysis.

## 🏆 FIDL Leaderboard 🏆

We make the FIDL leaderboard for unified ranking model's generalization across all domains. See [here](https://scu-zjz.github.io/ForensicHub-doc/rank/fidl_rank.html) for more details.

<div align="center">

| 🏆 Rank | Model | Deepfake 🖼️ | IMDL 📝 | AIGC 🤖 | Doc 📄 | Avg ⭐ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 1 | Effort | 0.614 | 0.587 | 0.410 | 0.788 | 0.600 |
| 🥈 2 | Clip-ViT-L/14 | 0.664 | 0.543 | 0.317 | 0.724 | 0.562 |
| 🥉 3 | UnivFD | 0.534 | 0.486 | 0.463 | 0.734 | 0.554 |
| 4 | ConvNeXT | 0.662 | 0.704 | 0.337 | 0.479 | 0.545 |
| 5 | Mesorch | 0.541 | 0.562 | 0.460 | 0.591 | 0.538 |
| 6 | IML-ViT | 0.581 | 0.562 | 0.325 | 0.626 | 0.523 |
| 7 | Segformer-b3 | 0.596 | 0.567 | 0.342 | 0.417 | 0.480 |
|   ...   |

</div>

## 🚤Update
- [2025.7.17] Released some missing pretrain weights for DocTamper Detection models, see this [issue](https://github.com/scu-zjz/ForensicHub/issues/9) for details.
- [2025.7.11] We update to a *lazy-load version* of MODEL and POSTFUNC. The package will be checked when the model is actually used, which reduces unnecessary package installation. 
- [2025.7.10] Add a script for single image inference, see [Code](https://github.com/scu-zjz/ForensicHub/blob/master/ForensicHub/training_scripts/inference.py).
- [2025.7.6] Add a new AIGC model, [FatFormer](https://arxiv.org/abs/2312.16649), see [Code](https://github.com/scu-zjz/ForensicHub/tree/master/ForensicHub/tasks/aigc/models/fatformer).
- [2025.7.1] Add document of Data Preparation & JSON Generation and Running Training & Evaluation in ForensicHub, see [Data Preparation](https://scu-zjz.github.io/ForensicHub-doc/guide/quickstart/3_data_preparation.html) and [Running Evaluation](https://scu-zjz.github.io/ForensicHub-doc/guide/quickstart/4_Running_Evaluation.html).
- [2025.6.22] Add summary of models and evaluators in ForensicHub, see [Document](https://scu-zjz.github.io/ForensicHub-doc/zh/summary/model.html).
- [2025.6.16] Add detailed installation and YAML configuration, see [Document](https://scu-zjz.github.io/ForensicHub-doc/zh/guide/quickstart/0_install.html).
- [2025.6.14] Add four new backbones: UNet, ViT, MobileNet, and DenseNet. More backbones are ongoing!


## 👨‍💻 About
☑️**About the Developers:**  
- ForensicHub's project leader/supervisor is Associate Professor 🏀[_Jizhe Zhou_ (周吉喆)](https://knightzjz.github.io/), Sichuan University🇨🇳, and _Jian Liu_ (刘健), the Leader of the Computer Vision Algorithm Research Group, Ant Group Company Ltd.   
- ForensicHub's codebase designer and coding leader is [_Bo Du_ (杜博)](https://github.com/dddb11), Sichuan University🇨🇳.  
- ForensicHub is jointly sponsored and advised by Prof. _Jiancheng LV_ (吕建成), Sichuan University 🐼, and Prof. _Chi-Man PUN_ (潘治文), University of Macau 🇲🇴, through the [Research Center of Machine Learning and Industry Intelligence, China MOE](https://center.dicalab.cn/) platform.  

## 📦 Resources
You can find the resources of models under IFF-Protocol, including [checkpoints](https://pan.baidu.com/s/1gER6MYt30ghrKQT0Nu182g?pwd=brir) (or [onedrive](https://1drv.ms/f/c/090693aab65eb63b/Eo6tl6ktl4BLkQjqjwqXRhYB5nNt_Sni5Nx6KMC4DDJPnw?e=vayq2E)), [training parameters](https://github.com/scu-zjz/ForensicHub/tree/master/ForensicHub/statics/crossdataset_image), and [hardware specifications](https://arxiv.org/pdf/2505.11003).

Checkpoints for Document Benchmark: https://pan.baidu.com/s/13ViyJebu12I0GN3BucBQrg?pwd=npkx or https://drive.google.com/drive/folders/1RZZxwYIX5e-lHKDw1CD45FwFC0QqJ7im?usp=sharing

Checkpoints for AIGC Benchmark: https://pan.baidu.com/s/11Jr2wjp6lAz9IBNWnbHlVg?pwd=kzhf or https://drive.google.com/drive/folders/1M-qe5xOblVZgKiBQ9j1Q-GQ4ao5VJMHZ?usp=sharing

Pretrained backbone weights for Document models: https://pan.baidu.com/s/1lsArVWzcJiADUcYYeqyClw?pwd=4gf4 or https://drive.google.com/drive/folders/1NiHeRAcG2VkoN-JFgV5O_4YynQFiQWUw?usp=sharing. Place the checkpoint under the corresponding model’s folder.

## 🕵️‍♂️ Architecture   
**ForensicHub provides four core modular components:** 

### 🗂️ Datasets

Datasets handle the data loading process and are required to return fields that conform to the ForensicHub
specification.

### 🔧 Transforms

Transforms handle the data pre-processing and augmentation for different tasks.

### 🧠 Models

Models, through alignment with Datasets and unified output, allow for the inclusion of various
state-of-the-art image forensic models.

### 📊 Evaluators

Evaluators cover commonly used image- and pixel-level metrics for different tasks, and are implemented with GPU
acceleration to improve evaluation efficiency during training and testing.

![](./images/overview.png)

## 📁 Project Structure Overview

```bash
ForensicHub/
├── common/                 # Common modules
│   ├── backbones/          # Backbones and feature extractors
│   ├── evalaution/         # Image- and pixel-level evaluators
│   ├── utils/              # Utilities
│   └── wrapper/            # Wrappers for dataset, model, etc.
├── core/                   # Core module providing abstract base classes
├── statics/                # YAML configuration files for training and testing
├── tasks/                  # Components for different sub-tasks
│   ├── aigc/           
│   ├── deepfake/             
│   ├── document/            
│   └── imdl/     
└── training_scripts        # Scripts for training and evaluation
```

## 📀Installation

---

We recommend cloning the project locally.

### 📉Clone

Simply run the following command:

```
git clone https://github.com/scu-zjz/ForensicHub.git
```
Also, since ForensicHub is compatible with DeepfakeBench (which hasn’t been uploaded to PyPI), you’ll need to clone our forked version [Site](https://github.com/scu-zjz/DeepfakeBench) locally and install it using: `pip install -e .`.

## 📝 YAML Configuration Guide

---

ForensicHub is entirely **configuration-driven**: you define your experiment in a single YAML file covering the model, datasets, transforms, evaluators, and all training hyperparameters. No code changes are needed to run different experiments.

### Execution Modes

ForensicHub supports three execution modes, set via the `flag` parameter:

| Mode | CLI Command | Description |
|------|-------------|-------------|
| `train` | `forhub train config.yaml` | Train a model with train/test datasets |
| `test` | `forhub test config.yaml` | Evaluate a trained model on test datasets |
| `run` | `forhub run config.yaml` | Run inference on images (prediction only, no ground truth needed) |

### Global Parameters

These parameters appear at the **root level** of the YAML file.

#### Execution & Hardware

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpus` | string | **required** | GPU IDs to use, e.g. `"0,1,2,3"` |
| `flag` | string | **required** | Execution mode: `train`, `test`, or `run` |
| `device` | string | `"cuda"` | Device: `"cuda"` or `"cpu"` |
| `seed` | integer | `42` | Random seed for reproducibility |
| `num_workers` | integer | `8` | Number of data loading workers |
| `pin_mem` | boolean | `true` | Pin memory for faster GPU data transfer |

#### Task Type

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `if_predict_label` | boolean | `false` | Model predicts image-level labels (classification) |
| `if_predict_mask` | boolean | `false` | Model predicts pixel-level masks (localization) |

> At least one must be `true`. Both can be `true` for models that do detection + localization.

#### Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | string | **required** | Directory for logs, checkpoints, and TensorBoard events |
| `log_per_epoch_count` | integer | `20` | Number of log entries printed per epoch |

#### Training Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | integer | - | Training batch size **per GPU** |
| `test_batch_size` | integer | - | Batch size for validation/testing |
| `epochs` | integer | - | Total number of training epochs |
| `accum_iter` | integer | `1` | Gradient accumulation steps (effective batch = `batch_size * num_gpus * accum_iter`) |
| `record_epoch` | integer | `0` | Only save best checkpoint after this epoch |
| `test_period` | integer | `1` | Run validation every N epochs |

#### Optimizer (AdamW)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | - | Learning rate. If not set, computed from `blr`: `lr = blr * effective_batch / 256` |
| `blr` | float | `0.1` | Base learning rate (used when `lr` is not specified) |
| `weight_decay` | float | `0.05` | L2 regularization weight |
| `min_lr` | float | - | Minimum learning rate for cosine annealing scheduler |
| `warmup_epochs` | integer | `1` | Number of linear warmup epochs before cosine decay |

> **LR Schedule**: Cosine annealing with linear warmup. Formula: `lr = min_lr + (lr - min_lr) * 0.5 * (1 + cos(pi * t / T))`

#### Distributed Training (DDP) & AMP

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `find_unused_parameters` | boolean | `true` | Allow unused parameters in DDP (needed for some models) |
| `use_amp` | boolean | `false` | Enable Automatic Mixed Precision for faster training |
| `world_size` | integer | `1` | Total number of distributed processes |
| `local_rank` | integer | `-1` | Local rank (auto-set by DDP launcher) |
| `dist_on_itp` | boolean | `false` | Use ITP cluster distributed init |
| `dist_url` | string | `"env://"` | URL for distributed process group initialization |

#### Checkpoint & Resume

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resume` | string | `""` | Path to a checkpoint to resume training from |
| `start_epoch` | integer | `0` | Starting epoch (auto-set when resuming) |
| `checkpoint_path` | string | - | Path to checkpoint for `test` and `run` modes |
| `no_model_eval` | boolean | `false` | If `true`, skip calling `model.eval()` during testing |

#### Inference-Only (flag: run)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_base_dir` | string | `"./run_output"` | Base output directory for predictions |
| `threshold` | float | - | Global prediction threshold (can be overridden per dataset) |

#### Miscellaneous

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_percentage` | float | - | Use only N% of the dataset (for ablation studies, value 0-100) |

---

### Model Configuration

The `model` block defines which model to use and its initialization parameters.

```yaml
model:
  name: <MODEL_NAME>
  post_func_name: <POST_FUNC>   # Optional: override post-processing function
  init_config:
    <model-specific parameters>
```

#### Available Models

<details>
<summary><b>AIGC Detection Models</b> (7 models)</summary>

| Name | Description | Key init_config |
|------|-------------|-----------------|
| `Dire` | DIRE — Diffusion-based reconstruction error | `guided_diffusion_path`, `model_path` |
| `HiFi_Net` | HiFi-Net — Hierarchical fine-grained detection | - |
| `DualNet` | DualNet — RGB + Noise dual-path network | - |
| `Synthbuster` | SynthBuster — Synthetic image detector | - |
| `UnivFD` | UnivFD — Universal fake detector (CLIP-based) | - |
| `FatFormer` | FatFormer — Frequency-aware transformer | - |
| `CO_SPY` | CO-SPY — Contrastive learning detector | - |

</details>

<details>
<summary><b>Document Manipulation Models</b> (6 models)</summary>

| Name | Description | Key init_config |
|------|-------------|-----------------|
| `DTD` | Document Tampering Detection | `convnext_path`, `swin_path` |
| `FFDN` | Fast Forensic Detection Network | `weight_path` (ConvNeXT pretrained) |
| `PSNet` | PSNet | - |
| `Tifdm` | TIFDM | - |
| `ADCDNet` | Attention-based Document Change Detection | `cls_n`, `loc_out_dim`, `dct_feat_dim`, `ce_weight`, `rec_weight`, ... |
| `CAFTB_Net` | CAFTB-Net (SegFormer-based) | - |

</details>

<details>
<summary><b>Deepfake Detection Models</b> (3 built-in + DeepfakeBench)</summary>

| Name | Description |
|------|-------------|
| `Spsl` | SPSL — Spatial Phase Spectrum Learning |
| `RecceDetector` | RECCE — Reconstruction-classification detector |
| `CapsuleNet` | Capsule Network for deepfakes |
| *DeepfakeBench models* | All models from [DeepfakeBench](https://github.com/scu-zjz/DeepfakeBench) are auto-wrapped (requires `pip install -e .` of the forked repo) |

</details>

<details>
<summary><b>Backbone / Generic Models</b> (11 backbones)</summary>

| Name | Description |
|------|-------------|
| `Resnet50` | ResNet-50 |
| `Resnet101` | ResNet-101 |
| `ConvNextSmall` | ConvNeXT Small |
| `ConvNextBase` | ConvNeXT Base |
| `Xception` | Xception |
| `EfficientNet` | EfficientNet |
| `DenseNet` | DenseNet |
| `MobileNet` | MobileNet |
| `ViT` | Vision Transformer |
| `UNet` | UNet (segmentation) |
| `SwinTransformer` | Swin Transformer |
| `SegFormer` | SegFormer (segmentation) |

Common `init_config` for backbones:
```yaml
init_config:
  pretrained: true
  num_classes: 1
  image_size: 256       # For models that need it
  output_type: "label"  # "label" or "mask"
```

</details>

<details>
<summary><b>Wrapper Models</b></summary>

| Name | Description |
|------|-------------|
| `Mask2LabelWrapper` | Wraps a mask-prediction model to produce image-level labels |

```yaml
model:
  name: Mask2LabelWrapper
  init_config:
    name: MVSSNet          # The underlying mask model
    init_config:           # Optional: params for the underlying model
      edge_lambda: 20
```

</details>

---

### Dataset Configuration

#### Training Dataset (`train_dataset`)

Can be a **single dataset** or a **list** (automatically concatenated).

```yaml
# Single dataset
train_dataset:
  name: <DATASET_NAME>
  dataset_name: <friendly_name>   # Used in logs
  init_config:
    <dataset-specific params>

# Multiple datasets (concatenated)
train_dataset:
  - name: DocDataset
    dataset_name: DocTamper_train
    init_config: { path: /data/doctamper, get_dct_qtb: true }
  - name: DocDataset
    dataset_name: DocTamper_scd_train
    init_config: { path: /data/scd, get_dct_qtb: true }
```

#### Test Dataset (`test_dataset`)

Always a **list**. Each entry can optionally override the global `evaluator`.

```yaml
test_dataset:
  - name: AIGCLabelDataset
    dataset_name: DiffusionForensics_val
    init_config:
      image_size: 224
      path: /data/val.json
  - name: AIGCLabelDataset
    dataset_name: ProGAN_val
    evaluator:                     # Per-dataset evaluator override
      - name: ImageAP
        init_config: { threshold: 0.5 }
    init_config:
      image_size: 224
      path: /data/progan_val.json
```

#### Run Dataset (`run_dataset`) — Inference mode only

```yaml
run_dataset:
  - name: DocDataset
    dataset_name: test_images
    output_dir: results_subdir      # Subdirectory under output_base_dir
    threshold: 0.5                  # Per-dataset threshold override
    init_config:
      path: /data/test_images
```

#### Available Datasets

<details>
<summary><b>All registered datasets</b></summary>

| Name | Task | Key init_config |
|------|------|-----------------|
| `AIGCLabelDataset` | AIGC | `path` (JSON), `image_size` |
| `AIGCCrossDataset` | AIGC | `path` (single or list of JSON paths), `image_size` |
| `DireDataset` | AIGC | `path`, `image_size` |
| `DocDataset` | Document | `path`, `get_dct_qtb` (bool), `dct_path`, `train` (bool), `suffix_img` (list) |
| `DocumentCrossDataset` | Document | `path` (list), `image_size`, `config_file`, `split_mode` |
| `DeepfakeCrossDataset` | Deepfake | `path`, `image_size`, `config_file`, `split_mode` (`train`/`val`/`test`) |
| `ManiDataset` | IMDL | `path`, `image_size` |
| `JsonDataset` | IMDL | `path` (JSON), `image_size`, `is_resizing`, `is_padding` |
| `BalancedDataset` | IMDL | `path`, `image_size` |
| `IMDLCrossDataset` | IMDL | `path`, `image_size` |

**Wrapper Datasets:**

| Name | Description | Key init_config |
|------|-------------|-----------------|
| `SlidingWindowWrapper` | Splits large images into overlapping patches | `patch_width`, `patch_height`, `overlapping` (0-1), `merge_mode`, `dataset` (nested) |
| `CrossDataset` | Combines multiple datasets with per-dataset sample limits | `dataset_config` (list with `name`, `pic_nums`, `init_config`) |

</details>

<details>
<summary><b>SlidingWindowWrapper example</b></summary>

```yaml
train_dataset:
  name: SlidingWindowWrapper
  dataset_name: doc_sliding
  init_config:
    patch_width: 512
    patch_height: 512
    overlapping: 0.5
    merge_mode: gaussian    # gaussian | mean | max | min | overwrite
    dataset:
      name: DocDataset
      init_config:
        path: /data/doc
        get_dct_qtb: true
```

</details>

<details>
<summary><b>CrossDataset example (multi-domain training)</b></summary>

```yaml
train_dataset:
  name: CrossDataset
  dataset_name: multi_domain
  init_config:
    dataset_config:
      - name: IMDLCrossDataset
        pic_nums: 12641
        init_config:
          image_size: 256
          path: /data/imdl
      - name: AIGCCrossDataset
        pic_nums: 12641
        init_config:
          image_size: 256
          path: /data/aigc
```

</details>

---

### Transform Configuration

```yaml
transform:
  name: <TRANSFORM_NAME>
  init_config:
    <transform-specific params>
```

| Name | Task | Key init_config |
|------|------|-----------------|
| `AIGCTransform` | AIGC | `norm_type` (`image_net`) |
| `DocTransform` | Document | `norm_type` (`image_net`), `compression_type` (`cv` / `pillow`), `luminance_path`, `chrominance_path` |
| `IMDLTransform` | IMDL | (task-specific) |

> All transforms use [Albumentations](https://albumentations.ai/) under the hood and return separate train/test/post-process pipelines.

---

### Evaluator Configuration

Evaluators compute metrics during training and testing. Defined globally or per test dataset.

```yaml
evaluator:
  - name: <EVALUATOR_NAME>
    init_config:
      threshold: 0.5
```

#### Available Evaluators

**Image-Level (classification):**

| Name | Metric |
|------|--------|
| `ImageF1` | F1 score |
| `ImageAUC` | Area Under ROC Curve |
| `ImageAP` | Average Precision |
| `ImageMCC` | Matthews Correlation Coefficient |
| `ImageTPR` | True Positive Rate (Recall) |
| `ImageTNR` | True Negative Rate (Specificity) |
| `ImageAccuracy` | Classification Accuracy |

**Pixel-Level (localization):**

| Name | Metric |
|------|--------|
| `PixelF1` | Pixel-wise F1 score |
| `PixelIOU` | Intersection over Union |
| `PixelMCC` | Pixel-level Matthews Correlation Coefficient |

> All evaluators are **GPU-accelerated** and accept a `threshold` parameter (default `0.5`). Additional pixel-level evaluators are available via the IMDLBenCo integration.

---

### Complete YAML Examples

<details>
<summary><b>Training — AIGC label prediction</b></summary>

```yaml
gpus: "0,1"
flag: train
log_dir: "./log/aigc_resnet_train"

if_predict_label: true
if_predict_mask: false

model:
  name: Resnet50
  init_config:
    pretrained: true
    num_classes: 1

train_dataset:
  name: AIGCLabelDataset
  dataset_name: DiffusionForensics_train
  init_config:
    image_size: 224
    path: /data/AIGC/DiffusionForensics/train.json

test_dataset:
  - name: AIGCLabelDataset
    dataset_name: DiffusionForensics_val
    init_config:
      image_size: 224
      path: /data/AIGC/DiffusionForensics/val.json

transform:
  name: AIGCTransform

evaluator:
  - name: ImageF1
    init_config:
      threshold: 0.5

batch_size: 768
test_batch_size: 128
epochs: 20
accum_iter: 1
record_epoch: 0

no_model_eval: false
test_period: 1
log_per_epoch_count: 20

find_unused_parameters: false
use_amp: true

weight_decay: 0.05
lr: 1e-4
blr: 0.001
min_lr: 1e-5
warmup_epochs: 1

device: "cuda"
seed: 42
resume: ""
start_epoch: 0
num_workers: 8
pin_mem: true
world_size: 1
local_rank: -1
dist_on_itp: false
dist_url: "env://"
```

</details>

<details>
<summary><b>Training — Document mask prediction</b></summary>

```yaml
gpus: "0,1,2,3"
flag: train
log_dir: "./log/doc_ffdn_train"

if_predict_label: false
if_predict_mask: true

model:
  name: FFDN
  init_config:
    weight_path: /weights/convnext_small.pth

train_dataset:
  name: DocDataset
  dataset_name: DocTamperData_train
  init_config:
    path: /data/DocTamper/train
    get_dct_qtb: true

test_dataset:
  - name: DocDataset
    dataset_name: DocTamperData_test
    init_config:
      path: /data/DocTamper/test
      get_dct_qtb: true

transform:
  name: DocTransform
  init_config:
    norm_type: image_net

evaluator:
  - name: PixelIOU
    init_config:
      threshold: 0.5
  - name: PixelF1
    init_config:
      threshold: 0.5

batch_size: 16
test_batch_size: 8
epochs: 100
accum_iter: 1
record_epoch: 0

no_model_eval: false
test_period: 1
log_per_epoch_count: 20

find_unused_parameters: true
use_amp: true

weight_decay: 0.05
lr: 1e-4
min_lr: 1e-5
warmup_epochs: 1

device: "cuda"
seed: 42
resume: ""
start_epoch: 0
num_workers: 8
pin_mem: true
dist_on_itp: false
dist_url: "env://"
```

</details>

<details>
<summary><b>Training — Mask2Label wrapper (use a mask model for classification)</b></summary>

```yaml
gpus: "0,1,2,3"
flag: train
log_dir: "./log/mask2label_mvssnet"

if_predict_label: true
if_predict_mask: false

model:
  name: Mask2LabelWrapper
  init_config:
    name: MVSSNet

train_dataset:
  name: AIGCLabelDataset
  dataset_name: DiffusionForensics_train
  init_config:
    image_size: 224
    path: /data/train.json

test_dataset:
  - name: AIGCLabelDataset
    dataset_name: DiffusionForensics_val
    init_config:
      image_size: 224
      path: /data/val.json

transform:
  name: AIGCTransform
  init_config:
    norm_type: image_net

evaluator:
  - name: ImageF1
    init_config:
      threshold: 0.5

batch_size: 128
test_batch_size: 32
epochs: 20
lr: 1e-4
weight_decay: 0.05
warmup_epochs: 1
use_amp: true
device: "cuda"
seed: 42
num_workers: 8
pin_mem: true
dist_url: "env://"
```

</details>

<details>
<summary><b>Inference — Run mode with SlidingWindowWrapper</b></summary>

```yaml
gpus: "0"
flag: run
log_dir: "./log/run_inference"

if_predict_label: false
if_predict_mask: true

output_base_dir: ./run_output
checkpoint_path: /weights/checkpoint-best.pth

model:
  name: FFDN
  init_config:
    weight_path: /weights/convnext_small.pth

run_dataset:
  - name: SlidingWindowWrapper
    dataset_name: fcd
    output_dir: fcd_output
    init_config:
      patch_width: 512
      patch_height: 512
      overlapping: 0.5
      merge_mode: gaussian
      dataset:
        name: DocDataset
        init_config:
          path: /data/fcd
          get_dct_qtb: true

transform:
  name: DocTransform
  init_config:
    norm_type: image_net

test_batch_size: 8
no_model_eval: false
device: "cuda"
num_workers: 8
pin_mem: true
dist_on_itp: false
dist_url: "env://"
```

</details>

<details>
<summary><b>Testing — Evaluate a checkpoint on multiple datasets</b></summary>

```yaml
gpus: "0"
flag: test
log_dir: "./log/eval_convnext"

if_predict_label: true
if_predict_mask: false

checkpoint_path: /weights/convnext_checkpoint.pth

model:
  name: ConvNextSmall
  init_config:
    image_size: 256

test_dataset:
  - name: AIGCLabelDataset
    dataset_name: DiffusionForensics_test
    init_config:
      image_size: 256
      path: /data/aigc_test.json
  - name: AIGCLabelDataset
    dataset_name: ProGAN_test
    evaluator:
      - name: ImageAP
        init_config:
          threshold: 0.5
    init_config:
      image_size: 256
      path: /data/progan_test.json

transform:
  name: AIGCTransform

evaluator:
  - name: ImageF1
    init_config:
      threshold: 0.5
  - name: ImageAUC
    init_config:
      threshold: 0.5

test_batch_size: 64
no_model_eval: false
device: "cuda"
num_workers: 8
pin_mem: true
dist_url: "env://"
```

</details>

---

### Running Experiments

```bash
# Training
forhub train /path/to/config.yaml

# Testing
forhub test /path/to/config.yaml

# Inference
forhub run /path/to/config.yaml

# Or via shell scripts
cd ForensicHub/statics
./run.sh                    # Single experiment (edit paths inside)
./batch_run.sh              # Batch experiments

# Single image inference (programmatic)
python ForensicHub/training_scripts/inference.py
```

> **Graceful stop**: Touch a file `STOP` in the `log_dir` during training to stop after the current epoch: `touch ./log/my_experiment/STOP`

---

## 🎯Quick Start

---

The Quick Start example is based on the local clone setup. ForensicHub is a modular and configuration-driven lightweight
framework. You only need to use the built-in or custom Dataset, Transform, and Model components, register them, and then
launch the pipeline using a YAML configuration file.

<details>
<summary>Training on the DiffusionForensics dataset using Resnet for AIGC</summary>

1. Dataset Preparation

Download the DiffusionForensics dataset from (https://github.com/ZhendongWang6/DIRE).
The experiment only uses the ImageNet portion. Format the data as JSON. ForensicHub does not restrict how the data is
loaded—just make sure the Dataset returns fields as defined in `\core\base_dataset.py`. This means users are free to
implement their own loading logic. In this case, we
use `/tasks/aigc/datasets/label_dataset.py`, which expects a JSON with entries like with label of 0 and 1 representing a
image of real and generated:

```
[
  {
    "path": "/mnt/data3/public_datasets/AIGC/DiffusionForensics/images/train/imagenet/real/n03982430/ILSVRC2012_val_00039791.JPEG",
    "label": 0
  },
  {
    "path": "/mnt/data3/public_datasets/AIGC/DiffusionForensics/images/train/imagenet/real/n03982430/ILSVRC2012_val_00022594.JPEG",
    "label": 0
  },
  ...
]
```

2. Component Preparation

In this example, the **Model** is ResNet50, which is already registered in `/common/backbones/resnet.py`, so no extra
code is needed. **Transform** is also pre-registered and available in `/tasks/aigc/transforms/aigc_transforms.py`,
providing basic
augmentations and ImageNet-standard normalization.

3. YAML Config & Training

ForensicHub supports lightweight configuration via YAML files. In this example, aside from data preparation, no
additional code is required.
Here is a sample training YAML `/statics/aigc/resnet_train.yaml`. The four components-**Model, Dataset, Transform,
Evaluator**-are all initiated
via `init_config`:

```yaml
# DDP
gpus: "4,5"
flag: train

# Log
log_dir: "./log/aigc_resnet_df_train"

# Task
if_predict_label: true
if_predict_mask: false

# Model
model:
  name: Resnet50
  init_config:
    pretrained: true
    num_classes: 1

# Train dataset
train_dataset:
  name: AIGCLabelDataset
  dataset_name: DiffusionForensics_train
  init_config:
    image_size: 224
    path: /data/AIGC/DiffusionForensics/train.json

# Test dataset (one or many)
test_dataset:
  - name: AIGCLabelDataset
    dataset_name: DiffusionForensics_val
    init_config:
      image_size: 224
      path: /data/AIGC/DiffusionForensics/val.json

# Transform
transform:
  name: AIGCTransform

# Evaluators
evaluator:
  - name: ImageF1
    init_config:
      threshold: 0.5

# Training hyperparameters
batch_size: 768
test_batch_size: 128
epochs: 20
accum_iter: 1
record_epoch: 0

no_model_eval: false
test_period: 1
log_per_epoch_count: 20

find_unused_parameters: false
use_amp: true

weight_decay: 0.05
lr: 1e-4
blr: 0.001
min_lr: 1e-5
warmup_epochs: 1

device: "cuda"
seed: 42
resume: ""
start_epoch: 0
num_workers: 8
pin_mem: true
world_size: 1
local_rank: -1
dist_on_itp: false
dist_url: "env://"
```

After creating the YAML file, you can launch training using `statics/run.sh` after updating file paths. You can also
use `statics/batch_run.sh` for batch experiments, which internally invokes multiple `run.sh` scripts. Testing works
similarly and only requires configuring the same four components.

</details>





## Citation

```
@misc{du2025forensichubunifiedbenchmark,
      title={ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization}, 
      author={Bo Du and Xuekang Zhu and Xiaochen Ma and Chenfan Qu and Kaiwen Feng and Zhe Yang and Chi-Man Pun and Jian Liu and Jizhe Zhou},
      year={2025},
      eprint={2505.11003},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.11003}, 
}
```
