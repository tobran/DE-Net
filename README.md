# DE-Net

Official Pytorch implementation for our paper [DE-Net: Dynamic Text-guided Image Editing Adversarial Networks](https://arxiv.org/pdf/2206.01160.pdf) by [Ming Tao](https://scholar.google.com/citations?user=5GlOlNUAAAAJ=en), [Bing-Kun Bao](https://scholar.google.com/citations?user=lDppvmoAAAAJ&hl=en), [Hao Tang](https://scholar.google.com/citations?user=9zJkeEMAAAAJ&hl=en), [Fei Wu](https://scholar.google.com/citations?user=tgeCjhEAAAAJ&hl=en), [Longhui Wei](https://scholar.google.com/citations?hl=en&user=thhnAhIAAAAJ), [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ). 

# Samples
<img src="results.jpg" width="877px" height="379px"/>

---

<img src="frame.jpeg" width="952px" height="380px"/>
---
## Requirements
- python 3.8
- Pytorch 1.9
- At least 1x12GB NVIDIA GPU
## Installation

Clone this repo.
```
git clone https://github.com/tobran/DE-Net
pip install -r requirements.txt
cd DE-Net/code/
```

## Preparation
### Datasets
1. Download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco2014](http://cocodataset.org/#download) dataset and extract the images to `data/coco/images/`


## Training
  ```
  cd DE-Net/code/
  ```
### Train the DE-Net model
  - For bird dataset: `bash scripts/train.sh ./cfg/bird.yml`
  - For coco dataset: `bash scripts/train.sh ./cfg/coco.yml`
### Resume training process
If your training process is interrupted unexpectedly, set **resume_epoch** and **resume_model_path** in train.sh to resume training.

