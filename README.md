# DeepCaption

DeepCaption is a framework for image captioning research using deep learning.  The code is based on the image captioning [tutorial by yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning) but has been extensively expanded since then.

The goal of image captioning is to convert a given input image into a natural language description. The encoder-decoder framework is widely used for this task. The image encoder is a convolutional neural network (CNN). Baseline code uses [resnet-152](https://arxiv.org/abs/1512.03385) model pretrained on the [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/) image classification dataset. The decoder is a long short-term memory (LSTM) recurrent neural network. 

## Features

DeepCaption supports many features, including:

- external pre-calculated features stored in numpy, lmdb or PicSOM bin format
- persistent features (features input at each RNN iteration)
- soft attention
- [teacher forcing scheduling](features.md#teacher-forcing-scheduling)

Some of the advanced features are documented on the [separate features documentation page](features.md).


## Usage 

### 1. Clone the repository

To get the latest release:

```bash
git clone https://github.com/aalto-cbir/DeepCaption
```

or to get the internal development version:

```bash
git clone https://version.aalto.fi/gitlab/CBIR/DeepCaption.git
```

### 2. Setup dataset for training

For example if you have downloaded the [COCO dataset](http://cocodataset.org/#download), you might have the images under /path/to/coco/images and annotations in /path/to/coco/annotations.

First we resize the images to 256x256.  This is just to speed up the training process.

```bash
./resize.py --image_dir /path/to/coco/images/train2014 --output_dir /path/to/coco/images/train2014_256x256
./resize.py --image_dir /path/to/coco/images/val2014 --output_dir /path/to/coco/images/val2014_256x256
```

Next, we need to set up the dataset configuration.  Create a file `datasets/datasets.conf` with the following contents:

```INI
[coco]
dataset_class = CocoDataset
root_dir = /path/to/coco

[coco:train2014]
image_dir = images/train2014_256x256
caption_path = annotations/captions_train2014.json

[coco:val2014]
image_dir = images/val2014_256x256
caption_path = annotations/captions_val2014.json
```

Now we can build the vocabulary:

```bash
./build_vocab.py --dataset coco:train2014 --vocab_output_path vocab.pkl
```

### 3. Train a model

Example of training a single model with default parameters on COCO dataset:

```bash
./train.py --dataset coco:train2014 --vocab vocab.pkl --model_name mymodel
```

or if you wish to follow validation set metrics:

```bash
./train.py --dataset coco:train2014 --vocab vocab.pkl --model_name mymodel --validate coco:val2014 --validation_scoring cider
```

You can plot the training and validation loss and other statistics using the following command:

```bash
./plot_stats.py models/mymodel/train_stats.json
```

By adding `--watch` you can have it update the plot automatically every time there are new numbers (typically after each epoch).


### 4. Infer from your model

Now you can use your model to generate a caption to any random image:

```bash
./infer.py --model models/mymodel/ep5.model --print_results random_image.jpg
```

or a directory of any random images:

```bash
./infer.py --model models/mymodel/ep5.model --print_results --image_dir random_image_dir/
```

You can also do inference on any configured dataset:

```bash
./infer.py --model models/mymodel/ep5.model --dataset coco:val2014
```

You can add e.g., `--validation_scoring cider` to automatically calculate scoring metrics if a ground truth has been defined for that dataset.

Inference also supports the following flags:
* `--max_seq_length` - maximum length of decoded caption (in words)
* `--no_repeat_sentences` - remove repeating sentences if they occur immediately after each other
* `--only_complete_senteces` - remove the last sentence if it does not end with a period (and thus is likely to be truncated)
