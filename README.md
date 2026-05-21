<h1 align="center">💥 MEGADance</h1>
<h3 align="center">Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2505.17543">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" alt="Paper">
  </a>
  <a href="https://sun-happy-ykx.github.io/MEGADance/">
    <img src="https://img.shields.io/badge/Project-Page-2ea44f?logo=githubpages&logoColor=white" alt="Project Page">
  </a>
  <a href="#training">
    <img src="https://img.shields.io/badge/Training-Code-4169e1?logo=github&logoColor=white" alt="Training Code">
  </a>
</p>

✨Training code release.✨

https://github.com/user-attachments/assets/0896dc29-0efa-41de-96ab-5c9823e013d9

> Music-driven 3D dance generation has attracted increasing attention in recent years, with promising applications in choreography, virtual reality, and creative content creation. Previous research has generated promising realistic dance movement from audio signals. However, traditional methods underutilize genre conditioning, often treating it as auxiliary modifiers rather than core semantic drivers. This oversight compromises music-motion synchronization and disrupts dance genre continuity, particularly during complex rhythmic transitions, thereby leading to visually unsatisfactory effects. To address the challenge, we propose MEGADance, a novel architecture for music-driven 3D dance generation. By decoupling choreographic consistency into dance generality and genre specificity, MEGADance demonstrates significant dance quality and strong genre controllability. It consists of two stages: (1) High-Fidelity Dance Quantization Stage (HFDQ), which encodes dance motions into a latent representation by Finite Scalar Quantization (FSQ) and reconstructs them with kinematic-dynamic constraints, and (2) Genre-Aware Dance Generation Stage (GADG), which maps music into the latent representation by synergistic utilization of Mixture-of-Experts (MoE) mechanism with Mamba-Transformer hybrid backbone. Extensive experiments on the FineDance and AIST++ dataset demonstrate the state-of-the-art performance of MEGADance both qualitatively and quantitatively.
---

## 🚀 Setup and Usage

### 🛠️ Set up the Environment

To set up the necessary environment for running this project, follow the steps below:

1. **Create a new conda environment**

   ```bash
   conda create -n MEGA_env python=3.10
   conda activate MEGA_env
   ```

2. **Install PyTorch (CUDA 12.8)**

   MEGADance requires **PyTorch 2.7.1 with CUDA 12.8**, which is available only through the official PyTorch wheel index:

   ```bash
   pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
       --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Install remaining dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 📦 Download Resources

- Download the **Preprocessed feature** from [Google Drive](https://drive.google.com/file/d/1Ttz28v_cgW3Fevu_kMquLBfN6BWdDwuI/view?usp=sharing) and place them into `./data/` folder.
- Download our **Pretrained model weights** and place them into the `./Pretrained/` folder:  
  [Download Link](https://drive.google.com/file/d/1lkCLmiD_4V1vaF8zkWFu8BUZ0COMoE5r/view?usp=sharing)
- Download the **Checkpoints for evaluation** and place them into the `./output/` folder:  
  [Download Link](https://drive.google.com/file/d/1PHDHvQjWasKYdy--Ge726TKbR7lg0CmW/view?usp=sharing)

---

## 🧩 Directory Structure

After downloading the necessary data and models, ensure the directory structure follows the pattern below:

```text
MEGADance/
|-- config/
|-- data/
|-- demo/
|-- models/
|-- output/
|-- Pretrained/
|-- utils/
|-- requirements.txt
|-- demo_gpt.py
|-- test_cls.py
|-- test_fsq.py
`-- test_gpt.py
```

---

<a id="training"></a>

## 🏋️ Training

### 🧪 Train the Model
To train the dance genre classifier:

```bash
python train_cls.py
```

To train the Stage1 model:

```bash
python train_fsq.py
```

To train the Stage2 model:

```bash
python train_gpt.py
```

---

## 📏 Evaluation

### 🧪 Evaluate the Model

To evaluate the Stage1 model's performance:

```bash
python test_fsq.py
```

To evaluate the Stage2 model's performance:

```bash
python test_gpt.py
```

## 🎬 Inference

To generate different genres dance based on a given music clip:

```bash
python demo_gpt.py --root_dir ./demo/1
```

This will generate the dance motion corresponding to the given music.


---

## 📄 Citation

```bibtex
@article{yang2025megadance,
  title={Megadance: Mixture-of-experts architecture for genre-aware 3d dance generation},
  author={Yang, Kaixing and Tang, Xulong and Peng, Ziqiao and Hu, Yuxuan and He, Jun and Liu, Hongyan},
  journal={arXiv preprint arXiv:2505.17543},
  year={2025}
}
```
