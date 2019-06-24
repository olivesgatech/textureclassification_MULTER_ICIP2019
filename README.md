# multer_dnn_textureclassification
Code for ["Multi-level Texture Encoding and Representation (MuLTER) based on Deep Neural Networks" (ICIP2019)](https://arxiv.org/abs/1905.09907);
Partial code are borrowed from Hang Zhang's work [deep texture encoding network (DEEP TEN)](https://github.com/zhanghang1989/PyTorch-Encoding).

## Citation
```
Y. Hu, Z. Long, and G. AlRegib, “Multi-level Texture Encoding and Representation (MuLTER) based on Deep Neural Networks,” IEEE International Conference on Image Processing, Taipei, Taiwan, September 2019.

or 

@inproceedings{hu2019_multertexture,
author={Hu, Yuting and Long, Zhiling and AlRegib, Ghassan},
booktitle={The 42nd IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), New Orleans, USA},
title={Multi-level Texture Encoding and Representation (MuLTER) based on Deep Neural Networks,
month={September},
year={2019}
}

```

## Prerequisites
```
Ubuntu 18.04
Python 3.6.6
Pytorchnightly 1.0.0.dev20190114

```

## Github repo
```
git clone https://github.com/yutinghu/deepten_multiscale.git
```

## Folder structure
The folder structure is listed as follows:
```
multer_dnn_textureclassification
    |----lib
        |----cpu
        |----gpu
        |----__pycache__
        |----__init__
    |----data
        |----minc-2500
            |----images
            |----labels
            |----categories.txt
    |----model
        |----resnet50-25c4b509.pth
    |----logs
        |----e.g. 20190623_185030
    |----main.py
    |----deepTEN.py
```

## Installations
We include a "lib" folder for pytorch encoding. You can also install pytorch encoding referring to [this link](https://hangzhang.org/PyTorch-Encoding/notes/compile.html).


## Data and model preparation
Download the [MINC-2500 dataset](http://opensurfaces.cs.cornell.edu/publications/minc/), which is a popular dataset for texture and material recognition.
Please unzip this file in the directory of `data/`. Download [ResNet-50](https://hangzh.s3.amazonaws.com/encoding/models/resnet50-25c4b509.zip) and unzip it in the "models" folder. 


## How to train your own model on minc-2500 dataset

Train:
```
$ CUDA_VISIBLE_DEVICES=0 python main.py --dataset minc --model deepten --batch-size 32 --lr 0.01 --epochs 30 --lr-step 10 --lr-scheduler step --weight-decay 5e-4

```
## How to test your own model on minc-2500 dataset
You can save a trained model (.pth file) to the "models" folder by adding this line (torch.save(model.state_dict(), PATH)) in the main.py.
Then, test:
```
python main.py --dataset minc --model name_of_pretrainedmodel --nclass 23  --pretrained --eval
```
