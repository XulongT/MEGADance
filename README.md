# MEGADance

Code for **NeurIPS 2025** paper  
**"MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation"**

[[Paper]]([https://arxiv.org/abs/2505.17543]) 

---

# Code

## Set up the Environment

To set up the necessary environment for running this project, follow the steps below:

1. **Create a new conda environment**

   ```bash
   conda create -n MEGA_env python=3.10
   conda activate MEGA_env
   ```

2. **Install PyTorch (CUDA 12.8)**

   MEGADance requires **PyTorch 2.7.1 with CUDA 12.8**, which is available only through the official PyTorch wheel index:

   ```
   pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
       --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Install remaining dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Download Resources

- Download the **Preprocessed feature** from [Google Drive](https://drive.google.com/file/d/1WBHuHvkjQKlWJYP2fVMEMT4NYBnXFxE7/view?usp=sharing) and place them into `./data/` folder.
- Download our **Pretrained model weights** and place them into the `./Pretrained/` folder:  
  [Download Link](https://drive.google.com/file/d/1rKTbH62v994UxAIHBmsOvoTjKBZUNZgq/view?usp=sharing)
- Download the **Checkpoints for evaluation** and place them into the `./output/` folder:  
  [Download Link](https://drive.google.com/placeholder)

---

## Directory Structure

After downloading the necessary data and models, ensure the directory structure follows the pattern below:

```
MEGADance/
    │
    ├── config/                   
    ├── data/                 
    ├── demo/             
    ├── models/                               
    ├── output/  
    ├── Pretrained/
    ├── utils/
    ├── requirements.txt
    ├── demo_gpt.py  
    ├── test_cls.py
    ├── test_fsq.py
    └── test_gpt.py     
```
---

## Training

Training code and instructions will be released soon.  
**Coming soon...**

---

## Evaluation

### 1. Generate Dance Sequences

To generate different genres dance based on a given music clip:

```bash
python demo_gpt.py --root_dir ./demo/1
```

This will generate the dance motion corresponding to the given music.

### 2. Evaluate the Model

To evaluate the Stage1 model’s performance:

```bash
python test_fsq.py
```

To evaluate the Stage2 model’s performance:

```bash
python test_gpt.py
```


---

# Citation

```bibtex
@article{yang2025megadance,
  title={Megadance: Mixture-of-experts architecture for genre-aware 3d dance generation},
  author={Yang, Kaixing and Tang, Xulong and Peng, Ziqiao and Hu, Yuxuan and He, Jun and Liu, Hongyan},
  journal={arXiv preprint arXiv:2505.17543},
  year={2025}
}
```
