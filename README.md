# Image captioning

Simple baseline image captioning based on [tutorial by yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

The goal of image captioning is to convert a given input image into a natural language description. The encoder-decoder framework is widely used for this task. The image encoder is a convolutional neural network (CNN). Baseline code uses [resnet-152](https://arxiv.org/abs/1512.03385) model pretrained on the [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/) image classification dataset. The decoder is a long short-term memory (LSTM) network. 

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

#### VisualGenome - Paragraph Captioning

```bash
$ python resize.py --image_dir datasets/data/VisualGenome/1.2/VG/1.2/images --output_dir datasets/processed/VisualGenome/resized_im2p_train --create_zip --subset datasets/processed/VisualGenome/im2p_train_split
```

#### VIST

```bash
$ python build_vocab.py --dataset vist --caption_path datasets/data/vist/dii/train.description-in-isolation.json --vocab_path datasets/processed/vist/vocab.pkl 
$ python resize.py --image_dir datasets/data/vist/images/train_full --output_dir datasets/processed/vist/train_full_resized --create_zip
```

### 4. Train the model

Example of training a single model with default parameters on COCO dataset:

```bash
$ python train.py --dataset coco --image_dir path/to/coco/resized.zip --caption_path datasets/data/COCO/annotations/captions_train2014.json --vocab_path datasets/processed/COCO/vocab.pkl --model_basename model-coco
```

Example of finetuning an existing model (trained on COCO) with a new dataset - VisualGenome Paragraph captions:

```bash
$ python train.py --dataset vgim2p --load_model models/run1/model-es256-hs512-nl2-bs128-lr0.001-da0.2-ep10.ckpt --num_epics 15 --model_basename coco-vg_im2p
```

It is also possible to train multiple models in multi-node environments using `scripts/multi_train.sh` helper script. This script takes as an argument a path to a CSV file with combinations of command line parameters. Example of such file available at [scripts/input/train_params.csv](train_params.csv). The header row in the CSV corresponds to the names of the command line parameters, each row corresponds to one model to be trained. Parameters other than the path to the CSV file are applied to each model being trained:

```bash
$ scripts/multi_train.sh scripts/input/train_params.csv --dataset coco --image_dir path/to/coco/resized.zip --caption_path datasets/data/COCO/annotations/captions_train2014.json --vocab_path datasets/processed/COCO/vocab.pkl --model_basename model-coco
```

### 5. Infer from model

As with training, it is possible to perform inference either on a single model:
```bash
$ python infer.py --image_dir datasets/data/COCO/images/val2014 --model models/model-coco-ep5.ckpt --vocab_path datasets/processed/COCO/vocab.pkl --verbose
```

Or with multiple models supplied as command line arguments to the helper script:

```bash
$ scripts/multi_infer.sh model1 model2 ... modelN --image_dir datasets/data/COCO/images/val2014 --vocab_path datasets/processed/COCO/vocab.pkl
```

Inference also supports the following flags:
* `--subset` - file containing list of image ids to include from the directory supplied in `--image_dir`
* `--max_seq_length` - maximum length of decoded caption (in words)
* `--no_repeat_sentences` - remove repeating sentences if they occur immediately after each other
* `--only_complete_senteces` - remove the last sentence if it does not end with a period (and thus is likely to be truncated)

### 6. Evaluate the model

#### COCO

COCO evaluation library works only with Python 2. Therefore you will need to make sure that you are running the below code in an environment that supports this.

1) Link to coco-caption library:

```bash
$ git clone https://github.com/tylin/coco-caption ~/workdir/coco-caption
$ cd image_captioning
$ ln -s path/to/coco-caption/pycocoevalcap datasets/
```

2) Install PyCocoTools for Python2 (your Python2 environment may requires loading modules):

```bash
$ module purge
$ module load python-env/2.7.10
$ pip2 install pycocotools --user
```

3) modify the file `pycocoevalcap/eval.py` to remove other metrics other than METEOR and CIDEr, on lines 39-45.

Once the setup steps above are done, you can perform evaluation on a json file that corresponds to one particular model:

```bash
$ python eval_coco.py path/to/result/captions.json --ground_truth datasets/data/COCO/annotations/captions_val2014.json
```
By default the above command creates a file containing METEOR and CIDEr score in JSON formatted file having an extension `*.eval`

If you want to evaluate multiple files at the same time, a helper script `scripts/multi_eval.sh` can be used:

```bash
$ scripts/multi_eval.sh coco2014 path/to/result_dir1/*.json path/to/result_dir2/*.json
```
The above command creates an `*.eval` files corresponding to each model being evaluated.

Finally, to simplify generating user readable output, an `eval2csv.py` script combines the produced `*.eval` files into a single, easy to parse and read CSV file:

```bash
$ python eval2csv.py --evaluations_dir path/containing/eval_files --output_file output_file.csv 
```

## Supported features - Training

### Teacher forcing scheduling

Teacher forcing in RNNs refers to feeding ground truth token at time step `t`. [This paper](https://arxiv.org/abs/1506.03099) details an approach where at each time step `t` ground truth is fed with probability `P(teacher_forcing)` and the token generated by the network itself with probability `1 - P(teacher_forcing)`. Currently inverse sigmoid sampling schedule is implemented for the above probabilities.

To train a model with sampled teacher forcing using `k=2200` and `beta=0.3` run the following:

```bash
$ python train.py --dataset coco:train2014 --model_name my_model --teacher_forcing sampled --teacher_forcing_k 2200 --teaacher_forcing_beta 0.3
```

This inverse sigmoid scheduling implementation depends on the parameter `k` which is usually in the order of `1000s` and can be interpreted as "how soon do we want to start decreasing the probability of teacher forcing?" and parameter `beta` which between `0` and `1` and can be interpreted as "once we start to use model's own outputs, how fast do we want the rate of model outputs usage to increase?", intuitively this is the slope of the middle segment of the inverse sigmoid curve.

Teacher forcing is controlled by parameter `--teacher_forcing`. By default this is set to `always`, meaning that we don't perform any sampling. Other options are `sampled` - using a sampling procedure outlined above; and `additive` - deterministic summation of teacher token with generated token with weights determined by the inverse sigmoid scheduler.

`--teacher_forcing_k` sets the value of `k` and `--teacher_forcing_beta` sets the value for `beta`.

# Feature Extraction

You can use `extract_dataset_features.py` to extract features from one of the convolutional models made available in `models.py`. Currently the following CNN models from PyTorch `torchvision` are supported `alexnet`, `Densenet 20`, `Resnet-152`, `VGG-16`, and `Inception V3`, all trained on ImageNet classification task. The exctracted features are either taken from the already flattened pre-classification layer, or by flattening the final convolutional or pooling layer.

The resulting features are saved using `lmdb` file format. Example command for generating features computed from images in MS-COCO training and validation sets using ResNet-152 CNN:

```bash
$ python extract_dataset_features.py --dataset coco:train2014+coco:val2014 --extractor resnet152
```

Feature extraction script currently supports the feature types specified by `--feature_type`:

* **plain** - takes an input image, resizes it and calculates features without any augmentation
* **avg** - takes 5 different crops of a resized input image - 4 corners + center, and then flips each crop horizontally, producing in total 10 cropped images. These images are then processed by the specified CNN separately, and the resulting single feature vector output produced by the feature extractor is formed by applying elementwise avareging over 10 feature vectors
* **max** - same as **avg**, but using elementwise maximum

Three different pixel value normalization strategies are currently supported for `avg` and `max` feature types. Normalization is specified by `--normalize` parameter:

* **default** - applies per-channel normalization settings [recommended](https://pytorch.org/docs/stable/torchvision/models.html) by PyTorch
* **skip** - do not normalize pixel values
* **substract_half** - subtract `0.5` from each pixel value, after the pixel values have been converted to be between `0` and `1`.

Feature extractor supports the same dataset configuration format as the `train.py` and `infer.py` scripts.

## DenseCap features

Dense Captioning features are extracted from DenseCap repository using LuA Torch.

### Installing LuA Torch

The following instructions are for CSC Taito cluster. Make sure that you run these commands in the GPU environment (interative shell) with `K80` GPU selected! (Running this on `P100` fails)

First, purge the environment load the needed modules:

`cd $USERAPPL  
module purge   
module load gcc/4.9.3 mkl/11.3.0 intelmpi/5.1.1 fftw/3.3.4 hdf5-serial/1.8.15 cuda/7.5`

Clone the LuA torch:

`git clone https://github.com/torch/distro.git ./torch --recursiv`

Compile and install:

`cd torch 
./clean.sh 
export CMAKE_LIBRARY_PATH=/appl/opt/mkl/11.3.0/compilers_and_libraries_2016.0.109/linux/mkl/lib/intel64_lin:/appl/opt/fftw/gcc-4.9.3/intelmpi-5.1.1/fftw-3.3.4/lib:/appl/vis/sox/14.4.2-n/lib 
export CMAKE_INCLUDE_PATH=/appl/opt/mkl/11.3.0/compilers_and_libraries_2016.0.109/linux/mkl/include:/appl/opt/fftw/gcc-4.9.3/intelmpi-5.1.1/fftw-3.3.4/include:/appl/vis/sox/14.4.2-n/include  
export CXX=g++  
export CC=gcc  
./install.sh`

Answer "NO" to the following question:

`Do you want to automatically prepend the Torch install location  
to PATH and LD_LIBRARY_PATH in your /homeappl/home/jppirhon/.bashrc? (yes/no)  
[yes] >>>  
no`

If all went well installation is now done.

You can now *test* the installation by first initializing the Torch environment:

`source $USERAPPL/torch/install/bin/torch-activate`

And starting Torch shell:

`th`

### Installing DenseCap model

Clone the DenseCap repository:

`cd $USERAPPL   
git clone https://github.com/jcjohnson/densecap`

Install the dependencies listed in the `README.md`

Fetch the pretrained model
`cd densecap   
 sh scripts/download_pretrained_model.sh`

Run a test command inside the `densecap` folder:

`th run_model.lua -input_image imgs/elephant.jpg`

If things were well, the `vis/data/` directory should have new `json` file with dense captioning output for the `elephant.jpg` image.

### Extracting DenseCap features

Before we are ready to extract the features we need to prepare a list of files containing the paths to the images that we need the features for. To do this, go to `image_captioning` directory and run the following script. Below is the example for MS COCO:

`python3 list_dataset_files.py --dataset coco:train2014:no_resize+coco:val2014:no_resize --num_workers 4`

In practice `--num_files 10` parameter can be used with the above command splits the file list into 10 files, to make it possible to parallelize DenseCap feature extraction

If your run the above command on Taito environment, it should have created a file:
`image_file_list-coco:train2014+coco:val2014-taito-gpu.csc.fi.txt` (the last part of file name will vary based on environment)

If all of this worked you are now ready to extract the features. Features are extracted using the handy `extract_features.lua` script provided in the repo. 

Now, to run feature extractor on a single file do the following:
`cd ../densecap   
th extract_features.lua -boxes_per_image 50 -input_txt ../image_captioning_dev/image_file_list-coco:train2014+coco:val2014-taito-gpu.csc.fi.txt -output_h5 densecap_features-coco:train2014+coco:val2014.h5`

The `extract_features.lua` script takes the following mandatory parameters:

* `-input_txt`  New line separated text file listing image paths to images for which we need to extract features
* `-output_h5` Path to HDF5 output file

The following parameters have default values, so they may not always be specified, however for our purposes some of them need to be changed:

* `-boxes_per_image` defaults to 100 - we can set this to *50* to match the papers we are replicating.
* `-gpu` defaults to 0, which GPU device to use

Other default parameters are:

* `-image_size` defaults to 720 - the dimension to which the image is resized before densecaptions are extracted keep it as it is, setting this to lower value may result in not enough regions being detected (LuA Torch model fails to handle these cases correctly)
* `-checkpoint` defaults to data/models/densecap/densecap-pretrained-vgg16.t7 which is a pretrained model we fetched earlier.  
* `-rpn_nms_thresh` defaults to 0.7 
* `-final_nms_thresh` defaults to 0.4
* `-num_proposals` defaults to 1000
* `-max_images` defaults to 0

#### Taito / CSC only

Doing this as SLURM batch/array job on 10 files on Taito would look like this:

Extract file list -

`python3 list_dataset_files.py --dataset coco:train2014:no_resize+coco:val2014:no_resize --num_workers 4 --num_files 10`

Run feature extraction as SLURM array job (please take note that the range of array job needs to be set to `0 to num_files - 1`):

`sbatch --time=0-24 --mem=128GB --job-name='COCO_TO_DENSECAP' --array=0-9 -o slurm-%x-%A_%a.out scripts/extract_densecap_features.sh
'../image_captioning_dev/file_lists/image_file_list-coco:train2014:no_resize+coco:val2014:no_resize-taito-gpu.csc.fi_${n}_of_${N}.txt' 
'../image_captioning_dev/features/densecap_features-coco:train2014:no_resize+coco:val2014:no_resize_${n}_of_${N}.h5`

Please look at `scripts/extract_densecap_features.sh` to see what the above command really does.

### Troubleshooting

If you get errors when running `th` commands, make sure you have first loaded the needed modules for LuA Torch (see above).
