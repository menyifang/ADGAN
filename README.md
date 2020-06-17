
# ADGAN
### [PyTorch](https://github.com/menyifang/ADGAN) | [project page](https://menyifang.github.io/projects/ADGAN/ADGAN.html) |   [paper](https://arxiv.org/pdf/2003.12267.pdf)

PyTorch implementation for controllable person image synthesis.

[Controllable Person Image Synthesis with Attribute-Decomposed GAN](https://menyifang.github.io/projects/ADGAN/ADGAN.html)  
 [Yifang Men](https://menyifang.github.io/),  [Yiming Mao](https://mtroym.github.io/), [Yuning Jiang](https://yuningjiang.github.io/), [Wei-Ying Ma](https://scholar.google.com/citations?user=SToCbu8AAAAJ&hl=en), [Zhouhui Lian](http://www.icst.pku.edu.cn/zlian/),
 Peking University & ByteDance AI Lab,
 **CVPR 2020(Oral)**.


**Component Attribute Transfer**
<p float="center">
<img src="gif/attributes.gif" width="800px"/>
</p>

**Pose Transfer**
<p float="center">
<img src="gif/pose.gif" width="800px"/>
</p>


## Requirement
* python 3
* pytorch(>=1.0)
* torchvision
* numpy
* scipy
* scikit-image
* pillow
* pandas
* tqdm
* dominate


## Getting Started

You can directly download our generated images (in Deepfashion) from [Google Drive](https://drive.google.com/drive/u/0/folders/1w1iF7UuI-drXZ1pNLcLg9pPT9Jmky9cy).

### Installation

- Clone this repo:
```bash
git clone https://github.com/menyifang/ADGAN.git
cd ADGAN
```

### Data Preperation
We use DeepFashion dataset and provide our **dataset split files**, **extracted keypoints files** and **extracted segmentation files** for convience.


The dataset structure is recommended as:
```
+—deepfashion
|   +—fashion_resize
|       +--train (files in 'train.lst')
|          +-- e.g. fashionMENDenimid0000008001_1front.jpg
|       +--test (files in 'test.lst')
|          +-- e.g. fashionMENDenimid0000056501_1front.jpg
|       +--trainK(keypoints of person images)
|          +-- e.g. fashionMENDenimid0000008001_1front.jpg.npy
|       +--testK
|          +-- e.g. fashionMENDenimid0000056501_1front.jpg.npy
|   +—semantic_merge
|   +—fashion-resize-pairs-train.csv
|   +—fashion-resize-pairs-test.csv
|   +—fashion-resize-annotation-pairs-train.csv
|   +—fashion-resize-annotation-pairs-test.csv
|   +—train.lst
|   +—test.lst
|   +—vgg19-dcbb9e9d.pth
|   +—vgg_conv.pth
...
```

1. Person images

<!-- - Download the DeepFashion dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) -->
- Download person images from [deep fasion dataset in-shop clothes retrival benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) and download dataset split from [Google Drive](https://drive.google.com/drive/u/0/folders/1bOjuIEp9KuV2wGk8XgCm35BNAfwZemrv).
- Crop the images. Split the raw images into the train split (```fashion_resize/train```) and the test split (```fashion_resize/test```). Launch
```bash
python tool/generate_fashion_datasets.py
``` 
**Note: In our settings, we crop the images of DeepFashion into the resolution of 176x256 in a center-crop manner.**

2. Keypoints files

- Download train/test pairs and train/test key points annotations from [Google Drive](https://drive.google.com/drive/u/0/folders/1bOjuIEp9KuV2wGk8XgCm35BNAfwZemrv), including **fashion-resize-pairs-train.csv**, **fashion-resize-pairs-test.csv**, **fashion-resize-annotation-train.csv**, **fashion-resize-annotation-train.csv**. Put these four files under the ```deepfashion``` directory.
- Generate the pose heatmaps. Launch
```bash
python tool/generate_pose_map_fashion.py
```

3. Segmentation files
- Extract human segmentation results from existing human parser (e.g. Look into Person) and merge into 8 categories. Our segmentation results are provided in [Google Drive](https://drive.google.com/drive/u/0/folders/1bOjuIEp9KuV2wGk8XgCm35BNAfwZemrv), including ‘semantic_merge2’ and ‘semantic_merge3’ in different merge manner. Put one of them under the ```deepfashion``` directory.



**Optionally, you can also generate these files by yourself.**

1. Keypoints files

We use [OpenPose](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) to generate keypoints. 

- Download pose estimator from [Google Drive](https://drive.google.com/drive/u/0/folders/1w1iF7UuI-drXZ1pNLcLg9pPT9Jmky9cy). Put it under the root folder ``ADGAN``.
- Change the paths **input_folder**  and **output_path** in ``tool/compute_coordinates.py``. And then launch
```bash
python2 compute_coordinates.py
```

2. Dataset split files

```bash
python2 tool/create_pairs_dataset.py
```

### Train a model

```bash
bash ./scripts/train.sh 
```

### Test a model
Download our pretrained model from [Google Drive](https://drive.google.com/drive/u/0/folders/1w1iF7UuI-drXZ1pNLcLg9pPT9Jmky9cy). Modify your data path and launch
```bash
bash ./scripts/test.sh 
```


### Evaluation
We adopt SSIM, IS, DS, CX for evaluation. This part is finished by [Yiming Mao](https://mtroym.github.io/). 

#### 1) SSIM

For evaluation, **Tensorflow 1.4.1(python3)** is required. 

```bash
python tool/getMetrics_market.py
```

#### 2) DS Score
Download pretrained on VOC 300x300 model and install propper caffe version [SSD](https://github.com/weiliu89/caffe/tree/ssd). Put it in the ssd_score forlder. 

```bash
python compute_ssd_score_fashion.py --input_dir path/to/generated/images
```

#### 3) CX (Contextual Score)

Refer to folder ‘cx’ to compute contextual score.




## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{men2020controllable,
  title={Controllable Person Image Synthesis with Attribute-Decomposed GAN},
  author={Men, Yifang and Mao, Yiming and Jiang, Yuning and Ma, Wei-Ying and Lian, Zhouhui},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2020 IEEE Conference on},
  year={2020}
}


```



## Acknowledgments
Our code is based on [PATN](https://github.com/tengteng95/Pose-Transfer) and thanks for their great work.

