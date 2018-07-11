# Image Segmentation Keras : Implementation of Segnet, FCN, UNet and other models in Keras.

Implememnation of various Deep Image Segmentation models in keras. 

<p align="center">
  <img src="https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/FCN1.png" width="50%" >
</p>

## Models 

* FCN8
* FCN32
* Simple Segnet
* VGG Segnet 
* U-Net
* VGG U-Net

## Getting Started

### Prerequisites

#### Basic Prerequisites
* Tensorflow
* Keras 2.0
* OpenCV for python
* Theano 

#### Conda Environment
You can use Conda to configure your environment. Conda file with all prerequisites are available [here](https://github.com/falreis/image-segmentation-keras/blob/master/i2dl.yml)

```shell
conda env create -f i2dl.yml
```

### Download the sample prepared dataset

Download and extract the following:

https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing

Place the dataset1/ folder in data/

## Visualizing the prepared data

You can also visualize your prepared annotations for verification of the prepared data.

```shell
python visualizeDataset.py \
 --images="data/dataset1/images_prepped_train/" \
 --annotations="data/dataset1/annotations_prepped_train/" \
 --n_classes=10 
```

## Downloading the Pretrained VGG Weights

You need to download the pretrained VGG-16 weights trained on imagenet if you want to use VGG based models

```shell
mkdir data
cd data
wget "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5"
```

## Training the Model

To train the model run the following command:

```shell
THEANO_FLAGS=device=gpu,floatX=float32  python  train.py \
 --save_weights_path=weights/ex1 \
 --train_images="data/dataset1/images_prepped_train/" \
 --train_annotations="data/dataset1/annotations_prepped_train/" \
 --val_images="data/dataset1/images_prepped_test/" \
 --val_annotations="data/dataset1/annotations_prepped_test/" \
 --n_classes=10 \
 --input_height=320 \
 --input_width=480 \
 --model_name="vgg_segnet" 
```
Choose model_name from segnet, vgg_segnet  vgg_unet, vgg_unet2, fcn8, fcn32

## Getting the predictions

To get the predictions of a trained model

```shell
THEANO_FLAGS=device=gpu,floatX=float32  python  predict.py \
 --save_weights_path=weights/ex1 \
 --epoch_number=0 \
 --test_images="data/dataset1/images_prepped_test/" \
 --output_path="data/predictions/" \
 --n_classes=10 \
 --input_height=320 \
 --input_width=480 \
 --model_name="vgg_segnet" 
```

