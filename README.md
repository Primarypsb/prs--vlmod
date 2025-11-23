
````markdown
<div align="center">

# VLMOD: Understanding Multi-Object World from Monocular View

> 🏆 **This repository is the entry for the [2025 VLP Challenge].**
> (本仓库为「2025 VLP 挑战赛」参赛作品)

[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-4b44ce.svg)](https://cvpr.thecvf.com/)
[![Challenge Track](https://img.shields.io/badge/VLP_Challenge-Track_B-orange)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](#-license)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)

<p align="center">
  <strong>Keyu Guo</strong>, <strong>Yongle Huang</strong>, <strong>Shijie Sun</strong>, <strong>Xiangyu Song</strong>, <strong>Mingtao Feng</strong>,<br>
  <strong>Zedong Liu</strong>, <strong>Huansheng Song</strong>, <strong>Tiantian Wang</strong>, <strong>Jianxin Li</strong>, <strong>Naveed Akhtar</strong>, <strong>Ajmal Saeed Mian</strong>
</p>

<br>

<img src="https://i.imgur.com/waxVImv.png" width="100%" alt="Teaser Image">

</div>

---

## 📢 News
* **[2025-02]** 🎉 Our paper has been accepted by **CVPR 2025**!
* **[2025-02]** We have released the code and pretrained weights for the **VLMOD Challenge (Track B)**.
* **[Repo]** Official Repository: [Primarypsb/prs-](https://github.com/Primarypsb/prs-)

---

## 📝 Introduction

This repository contains the official implementation for **VLMOD** (Track B), focusing on **Multi-Object 3D Visual Grounding (3DVG)** based on a **single monocular RGB image**.

### 🧠 Task Description
Given a monocular RGB image and a complex language description (e.g., *"find the red cup on the left side of the table and the black keyboard on the right side"*), the system predicts each referred object’s:
- **3D Position**: $(x, y, z)$
- **3D Size**: $(width, height, depth)$
- **Orientation**: Rotation angle

### 🚧 Core Challenges
- **Multi-object Scene Parsing**: Distinguishing multiple targets in cluttered environments.
- **Spatial Relationship Modeling**: Understanding relative positions (left, right, behind, etc.).
- **Accurate 3D Property Estimation**: Recovering depth and dimensions from a single 2D image.

---

## 📂 File Structure

The repository is organized as follows:

```text
/
│
├── 📜 data_utils.py         # Data parsing & feature statistics calculation
├── 📜 libs.py               # Core network modules (KAN, LatentParams, etc.)
├── 📜 model.py              # Architecture: TextEncoder & ObjectVAE
├── 📜 losses.py             # Loss functions: VAELoss, InfoNCELoss
├── 📜 train_dataset.py      # Dataset class (Loading, Jittering, Normalization)
│
├── 🚀 train.py              # Main training script
├── 🚀 run_inference.py      # Inference & Model Ensemble script
├── 📄 requirements.txt      # Python dependencies
│
├── 📁 train/                # [Data] Directory for Training JSON files
│   ├── 1632_..._obstacle.json
│   └── ...
│
├── 📁 test/                 # [Data] Directory for Testing JSON files
│   ├── 1632_..._obstacle.json
│   └── ...
│
├── 📁 downloaded-models/    # [Model] SBERT Pre-trained weights
│   └── 📁 all-MiniLM-L6-v2/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── model.safetensors
│
├── 📁 result/               # [Output] Inference results (Auto-generated)
│   ├── 1632_..._19_obstacle.txt
│   └── ...
│
├── 📦 feature_stats.pth     # [Checkpoints] Training data statistics (Mean/Std)
├── 📦 text_encoder.pth      # [Checkpoints] Trained Text Encoder weights
└── 📦 object_vae.pth        # [Checkpoints] Trained Object VAE weights
````

-----

## ⚙️ Installation & Setup

### 1\. Environment Requirements

We recommend using **Python \>= 3.9** (Developed with Python 3.12.3) and **PyTorch \>= 2.0.0**.

```bash
# Clone the repository
git clone [https://github.com/Primarypsb/prs-.git](https://github.com/Primarypsb/prs-.git)
cd prs-

# Install dependencies
pip install -r requirements.txt
```

### 2\. Download Models & Weights

You need to download the pretrained weights and place them in the correct directories.

| Model / File | Description | Download Link |
| :--- | :--- | :--- |
| **Full Checkpoints** | Includes `object_vae.pth`, `text_encoder.pth`, etc. | [Baidu Netdisk](https://pan.baidu.com/s/1LDGQdmlkgdxQL6_x55wDCw?pwd=1818) <br> **Code:** `1818` |

-----

## 🚀 Usage

### 1\. Data Preparation

Ensure your dataset JSON files are correctly placed in the respective folders:

  * **Training data** $\rightarrow$ `train/`
  * **Testing data** $\rightarrow$ `test/`

### 2\. Training

To train the model from scratch (this script calculates stats and trains the VAE/Encoder):

```bash
python train.py
```

> **Note:** This script will automatically generate `feature_stats.pth` and save the best model weights during the training process.

### 3\. Inference

To run inference on the test set or perform model ensemble:

```bash
python run_inference.py
```

The prediction results will be automatically saved in the `result/` directory.

-----

## 🤝 Contribution

We welcome contributions from the community\! You can:

  * 🛠️ **Reproduce** and verify the released modules.
  * ✨ **Implement** or improve other components.
  * 💡 **Contribute** new ideas for monocular 3D visual grounding.

-----

## 📜 License

This project is released under the **Apache 2.0 License** and is intended for academic and research purposes only.

-----

## 🏷️ Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{guo2025beyond,
  title={Beyond Human Perception: Understanding Multi-Object World from Monocular View},
  author={Guo, Keyu and Huang, Yongle and Sun, Shijie and Song, Xiangyu and Feng, Mingtao and Liu, Zedong and Song, Huansheng and Wang, Tiantian and Li, Jianxin and Akhtar, Naveed and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={3751--3760},
  year={2025}
}
```

```
```
