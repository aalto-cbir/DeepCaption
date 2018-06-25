Dataset configuration adapted from [Detectron](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/data/README.md)

# Setting Up Datasets

`datasets/data` directory contains symlinks to data locations.

## Creating Symlinks for COCO

Symlink the COCO dataset:

```
ln -s /path/to/coco $image_captioning/datasets/data/COCO
```

We assume that your local COCO dataset copy at `/path/to/coco` has the following directory structure:

```
COCO
|_ coco_train2014
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ coco_val2014
|_ ...
|_ annotations
   |_ instances_train2014.json
   |_ ...
```

If that is not the case, you may need to do something similar to:

```
mkdir -p $image_captioning/datasets/data/COCO
ln -s /path/to/coco_train2014 $image_captioning/datasets/data/COCO/
ln -s /path/to/coco_val2014 $image_captioning/datasets/data/COCO/
ln -s /path/to/json/annotations $image_captioning/datasets/data/COCO/annotations
```