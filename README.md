# Vision-Cognitive-Neural-Networks
This is the code base for **`"The potential of cognitive-inspired neural network modeling framework for computer vision processing tasks"`** submitted to Advanced Science in 2025.

## 🏠 TODOs

* [X] training and validation code.
* [-] Agri17K dataset (val).
* [] Agri17K dataset (train-val).
* [] Release the checkpoints.


## 🏠 Abstract
Visual computation models represented by deep neural networks (VDNNs) focus on replicating the selection of human visual attention, not the full spectrum of visual cognition, reflecting the divide between cognitive science (CS) and artificial intelligence (AI). To address this problem, we propose a cognitive modeling framework (CMF) that consists of three steps: abstracting cognitive function using functional, instantiating information transfer pipelines, and finally algorithmizing functional. Key challenges include modeling memory in VDNNs. We define memory in VDNNs as a priori information consisting of basic features in an image. We then introduce the Unbiased Mapping Algorithm (UMA), which uses the Fast Fourier Transform and statistical methods to model the priori information as long-term memory. Based on CMF and UMA, we develop the Visual Cognitive Neural Unit and Model, achieving top performance in various recognition tasks including natural scenes and smart agriculture. This work promotes the integration of AI and CS.


## 🏠 Overview
![figure1](figure/figure1.png)
![imagenet1k](figure/fiureg2.png)
![agri171k](figure/fiureg3.png)

## 🎁 Train and Test
We have provided detailed instructions for model training and testing and experimental details. 
### Install
- Clone this repo:

```bash
conda create -n dt python=3.10 -y
conda activate dt
git clone https://github.com/CAU-COE-VEICLab/Diffusion-Tuning.git
cd Diffusion-Tuning
```
- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```


### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

### Evaluation

To evaluate a pre-trained `Swin Transformer` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main_diffusion_tuning.py --eval \
--cfg <config-file> --pretrained <checkpoint> --data-path <imagenet-path> 
```

**Notes**:

- Please note that when testing the results of Diffusion Tuning (DT1~DT4), select the configuration file (--cfg) of the original model (the yaml file that does not contain 'dt').

For example, to evaluate the `Swin-B-DT1` with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 main_diffusion_tuning.py --eval \
--cfg configs/diffusion_finetune/swin/swin/swin_base_patch4_window7_224_22kto1k_finetune.yaml --pretrained dt1_swin_base_patch4_window7_224_22k.pth --data-path <imagenet-path>
```


## Practical Application

### AgriMaTech-LLaMA

To validate the effectiveness of diffusion tuning in specific tasks, we constructed a vertical domain supervised fine-tuning dataset for agricultural mechanization specialization - AgriMachine28K.  We then used LLaMA3.1-8B-Instruct as a base model and DT-3 as a training method on the AgriMachine28K and fine-tuning the base model to obtain AgriMaTech-LLaMA, a large language model in the field of assisted learning and teaching.  

you can click this [link](https://drive.google.com/drive/folders/1UYfqghaAWC0uqddyE6odlaGjKrjlsvQR?usp=drive_link) to download AgriMaTech-LLaMA.

**Notes**:

- The weights provided in the link are exclusively for LORA (Low-Rank Adaptation). You can load these weights using peft.PeftModel.from_pretrained provided by Hugging Face.

## Citation
