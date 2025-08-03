<div align="center">

<h1>The Potential of Cognitive-Inspired Neural Network Modeling Framework for Computer Vision</h1> 

<div>
    <a>Guorun Li</a>;
    <a>Lei Liu</a>;
    <a>Xiaoyu Li</a>;
    <a>Yuefeng Du*</a>;
</div>

<h3><strong>Accepted by Advanced Science in 2025</strong></h3>

</div>


## 🏠 TODOs

* [X] training and validation code.
* [X] Agri170K dataset (val).
* [ ] Agri170K dataset (train-val).
* [ ] Release the checkpoints.


## 🏠 Abstract
Vision deep neural networks (VDNNs) only simulate the attention-based significance selection function in human visual perception, rather than the full spectrum of visual cognition, reflecting the divide between cognitive science (CS) and artificial intelligence (AI). To address this problem, we propose a cognitive modeling framework (CMF) comprising three stages: functional abstraction, operator structuring, and program agent. Then, we define the prior information of basic image features as the long-term memory content in VDNNs. Meanwhile, we introduce a memory modeling method for VDNNs based on the fast Fourier transform (FFT) and statistical methods—the unbiased mapping algorithm (UMA). Finally, we develop visual cognitive neural units (VCNUs) and a baseline model (VCogM) based on CMF and UMA, and conduct performance testing experiments on different datasets such as natural scene recognition and agricultural image classification. The results show that VCogM and VCNU achieved state-of-the-art (SOTA) performance across various recognition tasks. The model’s learning process is independent of data distribution and scale, fully demonstrating the rationality of cognitive-inspired modeling principles. The research findings provide new insights into the deep integration of CS and AI.



## 🏠 Overview
![figure1](figure/CMF.jpg)
![VcogM](figure/VCogM.jpg)
![imagenet1k](figure/figure1.jpg)
![agri171k](figure/figure2.jpg)

## 🎁 Train and Test
We have provided detailed instructions for model training and testing and experimental details. 
### Install
- Clone this repo:

```bash
conda create -n vdnn python=3.10 -y
conda activate vdnn
git clone https://github.com/CAU-COE-VEICLab/Vision-Cognitive-Neural-Networks.git
cd Vision-Cognitive-Neural-Networks
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

To evaluate a pre-trained `VCogM` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main.py --eval \
--cfg <config-file, e.g.,  configs/sota_benchmark/vcnn/vcm_tiny_1k.yaml > --pretrained <checkpoint> --data-path <imagenet-path> 
```

To evaluate a pre-trained `VCogM` on Agri170K val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main_diffusion_tuning.py --eval \
--cfg <config-file, e.g., configs/vcnu_agri17k/vcnn/pretrain/vcm_tiny_agri17k.yaml> --pretrained <checkpoint> --data-path <imagenet-path> 
```

## Training from scratch 

To train the `VCogM-48M` on ImageNet1k, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main.py \
--cfg <config-file, e.g.,  configs/sota_benchmark/vcnn/vcm_small_1k.yaml > --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

To train the `VCogM-25M` on Agri170K, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main.py \
--cfg <config-file, e.g.,  configs/vcnu_agri17k/vcnn/pretrain/vcm_tiny_agri17k.yaml > --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

To train the `VCNU-21M` on Agri170K, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main.py \
--cfg <config-file, e.g.,  configs/vcnu_agri17k/vcnn/pretrain/vcnu_small_agri17k.yaml > --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

## Using UMA for your dataset

You can calculate the SSIM value of each image in your dataset by following this step: 
1. Using 'uma_tools/statistic_uma_strategy1.py' & 'uma_tools/statistic_uma_strategy2.py' to calculate the SSIM value of each image in the dataset.
   Then you can get the Excel file (named ssim_origin_excel_file), which contains the SSIM value of each image in your dataset 
2. Using 'uma_tools/count_frequency.py' to calculate the frequency distribution P in your dataset.

## Model Hub

TODOs
| name | pretrain | resolution |acc@1 |  #params | FLOPs | 1K model| Agri170K model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VCNU-T | ImageNet-1K | 224x224 | 78.2 | 13M | 2.3G | [baidu]()  | - |
| VCNU-S | ImageNet-1K | 224x224 | 80.8 |  21M | 4G | [baidu]()  | [baidu]()  |
| VCNU-B | ImageNet-1K | 224x224 | 81.8 |  37M | 6.8G | [baidu]() | - |
| VCogM-T | ImageNet-1K | 224x224 | 82.5 | 25M | 4.3G | [baidu]() | [baidu]()  |
| VCogM-S | ImageNet-1K | 224x224 | 83.9 |  48M | 8.7G | [baidu]()  | - |
| VCogM-B | ImageNet-1K | 224x224 | 84.4 |  92M | 17.1G | [baidu]()  | - |

## Agri170K dataset

We constructed a large-scale agricultural image dataset-Agri170K, comprising **96** categories and **173691** high-quality annotated images. 

These images cover various scenes, including fruits, animals, crops, and agricultural machinery. 

You can click this [link](https://drive.google.com/drive/folders/1L8yOT3EHHXxcVGlxBZwIIativjOw-r7X?usp=drive_link) to download Agri170K(train-val). 


## Acknowledge

Our implementations are partially inspired by [Swin Transformer](https://github.com/microsoft/Swin-Transformer).

Thanks for their great works!

## Citation
