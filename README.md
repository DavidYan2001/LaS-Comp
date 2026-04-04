<div align="center">

# [CVPR 2026] LaS-Comp: Zero-shot 3D Completion with Latent–Spatial Consistency

[![arXiv](https://img.shields.io/badge/arXiv-2602.18735-b31b1b.svg)](https://arxiv.org/abs/2602.18735)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/DavidYan2001/Omni-Comp3D)
[![Venue](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://cvpr.thecvf.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="assets/crop_logo.png" alt="LaS-Comp logo" width="280">

**Official implementation of "LaS-Comp: Zero-shot 3D Completion with Latent–Spatial Consistency"**

[Weilong Yan](https://davidyan2001.github.io/), [Haipeng Li](https://lhaippp.github.io/), [Hao Xu](https://hxwork.github.io/), [Nianjin Ye](https://scholar.google.com/citations?user=AhwGG78AAAAJ&hl=zh-CN), [Yihao Ai](https://ayh015-dev.github.io/), [Shuaicheng Liu](http://www.liushuaicheng.org/), [Jingyu Hu](https://scholar.google.com/citations?user=Gn0lRNsAAAAJ&hl=en)

</div>

---

## 📢 News
- **[2026-04]** 🎉 **LaS-Comp** is selected to do a poster presentation in [CHINA3DV 2026](http://china3dv.csig.org.cn/index.html)!
- **[2026-03]** 🎉 Our proposed benchmark **Omni-Comp3D** is now available at [Hugging Face](https://huggingface.co/datasets/DavidYan2001/Omni-Comp3D)!
- **[2026-02]** 📄 Our work is accepted by [CVPR 2026](https://cvpr.thecvf.com/Conferences/2026), and paper is available on [arXiv](https://arxiv.org/abs/2602.18735).

---

## ⚙️ Environment Setup

This project is tested under the following environment:

- **Python**: 3.10  
- **CUDA**: 12.1  
- **PyTorch**: 2.4.0 (**compiled with CUDA 12.1**)  
- **torchvision**: 0.19.0

> ⚠️ **Important:** Please ensure your system CUDA version is **12.1**.  
> Mismatched CUDA versions (e.g., 11.x or 12.3+) may cause errors with `spconv`, `flash-attn`, or rendering modules.

---
### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate lascomp
```

### 2. Install Python Dependencies
```
pip install -r requirements.txt
```

### 3. Verify Installation
```
python -c "import torch; print(torch.cuda.is_available())"
```

## ⚠️ Notes on Installation

### 1. CUDA Compatibility
This project relies on several CUDA-dependent libraries:

- `spconv-cu121`
- `flash-attn`
- `nvdiffrast`
- `diff-gaussian-rasterization`

### 2. Precompiled Packages

Some dependencies (e.g., xformers, kaolin) may require manual installation depending on your system.

## 📅 TODO
- [x] Release **Omni-Comp3D** dataset.
- [x] Release code for **TRELLIS**.
- [ ] Release code for **Direct3D-S2**.


## 📥 Checkpoint Download

Following the official [TRELLIS](https://github.com/microsoft/TRELLIS) release, please download the pretrained checkpoints for **`image-large`** and **`text-xlarge`**, and place them under the `ckpt/` directory. The directory structure should look like:

```
ckpt/
├── image-large/
├── text-xlarge/
└── clip/
```

## 📦 Dataset
The **Omni-Comp3D** benchmark is hosted on Hugging Face. It includes two parts:

- **Omni-Comp3D**: our proposed benchmark for comprehensive completion evaluation.
- **samples**: the evaluation samples following SDS-Complete, GenPC, ComPC.

You can access them here: 👉 [**Omni-Comp3D**](https://huggingface.co/datasets/DavidYan2001/Omni-Comp3D)

Please place both **Omni-Comp3D** and **samples** in their appropriate locations within the project directory.

## 🚀 Running the Project

For custom partial inputs, you can run the project in either **text-conditioned** or **image-conditioned** mode.

### 1. Text-conditioned completion

```
python run_lascomp_text_condition_single.py \
  --partial-path path/to/your/partial-shape \
  --prompt "Your prompt" \
  --dataset custom \
  --yz-flip
```
### 2. Image-conditioned completion
```
python run_lascomp_image_condition_single.py \
  --partial-path path/to/your/partial-shape \
  --image-path path/to/your/image \
  --dataset custom \
  --yz-flip
```
### Notes
- `--partial-path` specifies the input partial 3D shape.
- `--prompt` is used for the text-conditioned model.
- `--image-path` is used for the image-conditioned model.
- `--dataset custom` indicates that the input comes from your own custom data.
- `--yz-flip` is used for samples whose vertical axis is **y**. If your sample uses **z** as the default vertical axis, please use `--no-yz-flip` instead.
- You can adjust the hyperparameters to get better completion results.

### 3. Benchmark Evaluation

To evaluate the model on the provided benchmarks, you can use the following scripts for different datasets.

#### Omni-Comp3D
```
python run_lascomp_text_condition_omnicomp.py
```

#### Redwood and Synthetic
```
python run_lascomp_image_condition.py
```
or
```
python run_lascomp_text_condition.py
```
## 🙏 Acknowledgements

We sincerely thank [TRELLIS](https://github.com/microsoft/TRELLIS), [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2), [ComPC](https://github.com/Tianxinhuang/ComPC), [FlowChef](https://github.com/FlowChef/FlowChef), [FlowDPS](https://github.com/FlowDPS-Inverse/FlowDPS), and [VoxHammer](https://github.com/Nelipot-Lee/VoxHammer) for their inspirational help to our work.

## 📝 Citation
If you find our work or dataset helpful for your research, please consider citing:

```bibtex
@article{lascomp,
  title={La{S}-{C}omp: {Z}ero-shot 3{D} {C}ompletion with {L}atent-{S}patial {C}onsistency},
  author={Weilong Yan and Haipeng Li and Hao Xu and Nianjin Ye and Yihao Ai and Shuaicheng Liu and Jingyu Hu},
  journal={arXiv preprint arXiv:2602.18735},
  year={2026},
}
