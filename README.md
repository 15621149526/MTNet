[English](#) | [简体中文](#)

<div align="center">

## MTNet: A Robust Modulation Recognition Method Based on Mamba and Non-Local Memory Aggregation Transformer

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![stars](https://img.shields.io/github/stars/15621149526/MTNet?style=flat&color=orange)](https://github.com/15621149526/MTNet/stargazers)
[![issues](https://img.shields.io/github/issues/15621149526/MTNet?color=red)](https://github.com/15621149526/MTNet/issues)
[![paper](https://img.shields.io/badge/Paper-arXiv-B31B1B.svg)](https://arxiv.org/)
[![contact](https://img.shields.io/badge/Contact-Email-007EC6.svg)](mailto:233071203111@stu.sdmu.edu.cn)

</div>

This is the official PyTorch implementation for the paper:
> **[MTNet: A Robust Modulation Recognition Method Based on Mamba and Non-Local Memory Aggregation Transformer](#)**

### 🔥 News
- **[2026/03]** The core training code, dynamic genetic augmentation, and optimal pre-trained weights for `RadioML 2016.10a` and `RML22.01A` are now officially released!

### ✨ Key Features
- **Shallow Temporal Modeling:** Built upon the S6-based **Mamba** block for ultra-long IQ signal sequences representation with linear complexity.
- **Dynamic SoftThresholding:** A dual-statistical awareness denoising mechanism filtering environmental noise effectively.
- **NMDDA Block:** Non-local Memory Dense Dynamic Aggregation to achieve cross-layer feature fusion with error-gated selective retrieval.

### 📈 Performance
MTNet achieves state-of-the-art (SOTA) accuracy under low Signal-to-Noise Ratio (SNR) conditions across multiple standard datasets. 
*(Detailed visualization scripts for SNR curves and confusion matrices are included in this repository).*

### 📌 Citation
If you find our work, model architecture, or pre-trained weights useful for your research, please consider giving this repository a star ⭐ and citing our paper:

```bibtex
@misc{mtnet2024,
    title={MTNet: A Robust Modulation Recognition Method Based on Mamba and Non-Local Memory Aggregation Transformer},
    author={Your Name and Co-authors},
    year={2024},
    eprint={xxxx.xxxxx},
    archivePrefix={arXiv},
    primaryClass={eess.SP},
}
