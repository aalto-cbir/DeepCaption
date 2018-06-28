# Image captioning

Simple baseline image captioning based on [tutorial by yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

The goal of image captioning is to convert a given input image into a natural language description. The encoder-decoder framework is widely used for this task. The image encoder is a convolutional neural network (CNN). Baseline code uses [resnet-152](https://arxiv.org/abs/1512.03385) model pretrained on the [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/) image classification dataset. The decoder is a long short-term memory (LSTM) network. 

![alt text](png/model.png)

#### Training phase
For the encoder part, the pretrained CNN extracts the feature vector from a given input image. The feature vector is linearly transformed to have the same dimension as the input dimension of the LSTM network. For the decoder part, source and target texts are predefined. For example, if the image description is **"Giraffes standing next to each other"**, the source sequence is a list containing **['\<start\>', 'Giraffes', 'standing', 'next', 'to', 'each', 'other']** and the target sequence is a list containing **['Giraffes', 'standing', 'next', 'to', 'each', 'other', '\<end\>']**. Using these source and target sequences and the feature vector, the LSTM decoder is trained as a language model conditioned on the feature vector.

#### Test phase
In the test phase, the encoder part is almost same as the training phase. The only difference is that batchnorm layer uses moving average and variance instead of mini-batch statistics. This can be easily implemented using [encoder.eval()](sample.py#L37). For the decoder part, there is a significant difference between the training phase and the test phase. In the test phase, the LSTM decoder can't see the image description. To deal with this problem, the LSTM decoder feeds back the previosly generated word to the next input. This can be implemented using a [for-loop](model.py#L48).

## Usage 

### 1. Clone the repositories
```bash
$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI/
$ make
$ python setup.py build
$ python setup.py install
$ cd ../../
$ git clone git@version.aalto.fi:CBIR/image_captioning.git
$ cd image_captioning
```

### 2. Create links to the datasets you plan to use for training / evaluation

#### COCO

Link to COCO dataset:

```bash
$ ln -s /path/to/coco datasets/data/COCO
$ mkdir -p datasets/processed/COCO
```
#### VIST

Link to VIST dataset:

```bash
$ ln -s /path/to/vist datasets/data/vist
$ mkdir -p datasets/processed/vist
```


### 3. Preprocessing

Build vocabulary and resize images:

#### COCO

```bash
$ python build_vocab.py --dataset coco --caption_path datasets/data/COCO/annotations/captions_train2014.json --vocab_path datasets/processed/COCO/vocab.pkl 
$ python resize.py --image_dir datasets/data/COCO/train2014 --output_dir datasets/processed/COCO/train2014_resized --create_zip
```

_Note:_ An optional `--create_zip` parameter zips the resized images directory.

#### VIST

```bash
$ python build_vocab.py --dataset vist --caption_path datasets/data/vist/dii/train.description-in-isolation.json --vocab_path datasets/processed/vist/vocab.pkl 
$ python resize.py --image_dir datasets/data/vist/images/train_full --output_dir datasets/processed/vist/train_full_resized --create_zip
```

### 4. Train the model

#### COCO 

```bash
$ python train.py --dataset coco --image_dir path/to/coco/resized.zip --caption_path datasets/data/COCO/annotations/captions_train2014.json --vocab_path datasets/processed/COCO/vocab.pkl --model_basename model-coco
```

#### 5. Test the model 

```bash
$ python sample.py --image='png/example.png'
```

