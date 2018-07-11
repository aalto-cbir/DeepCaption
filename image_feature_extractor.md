### Info:
- Available extractors: resnet-152
- Features dimension: [1,2048] (without batch normalization), [1,256] (feature vector is linearly transformed to have the same dimension as the input dimension of the LSTM network)
- Required data: images

### Usage:
Invoke the `extract_image_features.sh` script the following way:

```
sbatch extract_image_features.sh environment image_folder_location output_folder_location resized_flag
```

Example:

```
sbatch extract_image_features.sh taito /proj/memad/COCO/resized_train2014/ ./features/ true
```

### Details:

- Rezizing of images will be skipped if the `resized_flag` passed is `true`
- Images location is defaulted to COCO images based on the environment if the `image_folder_location` is not passed
- Output features location is defaulted to `./features/` if the `output_folder_location` is not passed
- Valid environments that can be passed for now are `taito/triton`
- Features are saved as pickle (`.pkl`) files per image (image name is used as the name of the output file)

Other job trigger related details can be inferred from: https://version.aalto.fi/gitlab/CBIR/image_captioning/blob/image_feature_extractor/extract_image_features.sh