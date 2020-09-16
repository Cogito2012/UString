# UString
This repo contains code for our following paper:

Wentao Bao, Qi Yu and Yu Kong, Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning, submitted to ACM Multimedia 2020.

## Contents
0. [Overview](#overview)
0. [Dataset Preparation](#dataset)
0. [Pre-trained Models](#models)
0. [Installation Guide](#install)
0. [Train and Test](#traintest)
0. [Citation](#citation)

<a name="overview"></a>
## :globe_with_meridians:  Overview 
<div align=center>
  <img src="demo/000821_vis.gif" alt="Visualization Demo" width="800"/>
</div>

We propose an uncertainty-based traffic accident anticipation model for dashboard camera videos. The task aims to accurately identify traffic accidents and anticipate them as early as possible. We first use Cascade R-CNN to detect bounding boxes of each frame as risky region proposals. Then, the features of these proposals are fed into our model to predict accident scores (red curve). In the same time, both aleatoric (brown region) and epistemic (yellow region) uncertainties are predicted by Bayesian neural networks.

<a name="dataset"></a>
## :file_cabinet:  Dataset Preparation

The code currently supports three datasets., DAD, A3D, and CCD. These datasets need to be prepared under the folder `data/`. 

> * For CCD dataset, please refer to the [CarCrashDataset](https://github.com/Cogito2012/CarCrashDataset) repo for downloading and deployment. 
> * For DAD dataset, you can acquire it from [DAD official](https://github.com/smallcorgi/Anticipating-Accidents). The officially provided features are grouped into batches while it is more standard to split them into separate files for training and testing. To this end, you can use the script `./script/split_dad.py`. 
> * For A3D dataset, the annotations and videos are obtained from [A3D official](https://github.com/MoonBlvd/tad-IROS2019). Since it is sophiscated to process it for traffic accident anticipation with the same setting as DAD, you can directly download our processed A3D dataset from Google Drive: [A3D processed](https://drive.google.com/drive/folders/1loK_Cr1UHZGJpetUIQCSI3NlBQWynK3v?usp=sharing).

<a name="models"></a>
## :file_cabinet:  Pre-trained Models

Choose the following files according to your need.

> * [**Cascade R-CNN**](https://drive.google.com/drive/folders/1fbjKrzgXv_FobuIAS37k9beCkxYzVavi?usp=sharing): The pre-trained Cascade R-CNN model files and modified source files. Please download and extract them under `lib/mmdetection/`.
> * [**Pre-trained UString Models**](https://drive.google.com/drive/folders/1yUJnxwDtn2JGDOf_weVMDOwywdkWULG2?usp=sharing): The pretrained model weights for testing and demo usages. Download them and put them anywhere you like.

<a name="install"></a>
## :file_cabinet: Installation Guide

### 1. Setup Python Environment

The code is implemented with `Python=3.7.4` and `PyTorch=1.0.0` with `CUDA=10.0.130` and `cuDNN=7.6.3`. We highly recommend using Anaconda to create virtual environment to run this code. Please follow the following installation dependencies strictly:
```shell
# create python environment
conda create -n py37 python=3.7

# activate environment
conda activate py37

# install dependencies
pip install -r requirements.txt
```

### 2. Setup MMDetection Environment (Optional)

If you need to use mmdetection for training and testing Cascade R-CNN models, you may need to setup an mmdetection environment separately such as `mmlab`. Please follow the [official mmdetection installation guide](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md).
```shell
# create python environment
conda create -n mmlab python=3.7

# activate environment
conda activate mmlab

# install dependencies
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv==0.4.2

# Follow the instructions at https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v1.1.0  # important!
cp -r ../Cascade\ R-CNN/* ./  # copy the downloaded files into mmdetection folder

# compile & install
pip install -v -e .
python setup.py install

# Then you are all set!
```
**Note**: This repo currently does not support `CUDA>=10.2` environment, as the object detection API we used is no longer supported only by the latest `mmdetection`, and the `torch-geometry` lib we sued is dependent on `PyTorch=1.0`. We will release the support for the lastest CUDA and PyTorch. 

<a name="traintest"></a>
## :rocket: Train and Test

### 1. Demo

We provide an end-to-end demo to predict accident curves with given video. Note that before you run the following script, both the python and mmdetection environments above are needed. The following command is an example using the pretrained model on CCD dataset. The model file is placed at `demo/final_model_ccd.pth` by default.

```shell
bash run_demo.sh demo/000821.mp4
```
Results will be saved in the same folder `demo/`.

### 2. Test the pre-trained UString model

Take the DAD dataset as an example, after the DAD dataset is correctly configured, run the following command. By default the model file is placed at `output/UString/vgg16/snapshot/final_model.pth`.
```shell
# For dad dataset, use GPU_ID=0 and batch_size=10.
bash run_train_test.sh test 0 dad 10
```
The evaluation results on test set will be reported, and visualization results will be saved in `output/UString/vgg16/test/`.

### 3. Train UString from scratch.

To train UString model from scratch, run the following commands for DAD dataset:
```shell
# For dad dataset, use GPU_ID=0 and batch_size=10.
bash run_train_test.sh train 0 dad 10
```
By default, the snapshot of each checkpoint file will be saved in `output/UString/vgg16/snapshot/`.


<a name="citation"></a>
## :bookmark_tabs:  Citation

Please cite our paper if you find our code useful.

```
@InProceedings{BaoMM2020,
    author = {Bao, Wentao and Yu, Qi and Kong, Yu},
    title  = {Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning},
    booktitle = {Proceedings of the 28th ACM International Conference on Multimedia (MM ’20)},
    month  = {October},
    year   = {2020}
}
```

If you have any questions, please feel free to leave issues in this repo or contact [Wentao Bao](mailto:wb6219@rit.edu) by email. Note that part of codes in `src/` are referred from [VGRNN](https://github.com/VGraphRNN/VGRNN) project.
