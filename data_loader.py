import glob
import json
import nltk
import os
import re
import pickle
import sys
import zipfile

import numpy as np
import torch
import torch.utils.data as data

from collections import namedtuple
from PIL import Image
import configparser


def basename(fname):
    return os.path.splitext(os.path.basename(fname))[0]


DatasetConfig = namedtuple('DatasetConfig',
                           'name, child_split, dataset_class, image_dir, caption_path, '
                           'features_path, subset, config_dict')


class DatasetParams:
    def __init__(self, dataset_config_file=None):
        """Initialize dataset configuration object, by default loading data from
        datasets/datasets.conf file or creating a generic configuration if datasets.conf
        file was not specified or found"""

        self.config = configparser.ConfigParser()

        config_path = self._get_config_path(dataset_config_file)

        # Generic, "fall back" dataset
        self.config['generic'] = {'dataset_class': 'GenericDataset'}

        # If the configuration file is not found, we can still use
        # 'generic' dataset with sensible defaults when infering.
        if not config_path:
            print(('Config file <{}> not found. Loading default settings '+
                   'for generic dataset.').format(dataset_config_file))
            print('Hint: you can use datasets/datasets.conf.default '+
                  'as a starting point.')

        # Otherwise all is good, and we are using the config file as
        else:
            print("Loading dataset configuration from {}...".format(config_path))
            self.config.read(config_path)

    def _get_config_path(self, dataset_config_file):
        """Try to intelligently find the configuration file"""
        # List of places to look for dataset configuration - in this order:
        # In current working directory
        conf_in_working_dir = dataset_config_file
        # In user configuration directory:
        conf_in_user_dir = os.path.expanduser(os.path.join("~/.config/image_captioning",
                                                           dataset_config_file))
        # Inside code folder:
        file_path = os.path.realpath(__file__)
        conf_in_code_dir = os.path.join(os.path.dirname(file_path), dataset_config_file)

        search_paths = [conf_in_working_dir, conf_in_user_dir, conf_in_code_dir]

        config_path = None
        for i, path in enumerate(search_paths):
            if os.path.isfile(path):
                config_path = path
                break

        return config_path

    def get_params(self, args_dataset, image_dir=None, image_files=None):

        datasets = args_dataset.split('+')
        configs = []
        for dataset in datasets:
            dataset_name_orig = dataset  # needed if lower-casing is used
            # dataset = dataset.lower() # disabled temporarily

            if dataset not in self.config:
                print('No such dataset configured:', dataset)
                sys.exit(1)

            if self.config[dataset]:
                # If dataset is of the form "parent_dataset:child_split",
                # if child doesn't have a parameter specified use fallback values from parent
                cfg, child_split = self._combine_cfg(dataset)

                dataset_name = dataset
                dataset_class = cfg['dataset_class']

                root = None
                if dataset == 'generic' and (image_files or image_dir):
                    image_list = []
                    if image_files:
                        image_list += image_files
                    if image_dir:
                        image_list += glob.glob(image_dir + '/*.jpg')
                        image_list += glob.glob(image_dir + '/*.jpeg')
                        image_list += glob.glob(image_dir + '/*.png')
                    root = image_list
                else:
                    root = self._cfg_path(cfg, 'image_dir')

                caption_path = self._cfg_path(cfg, 'caption_path')
                features_path = self._cfg_path(cfg, 'features_path')
                subset = cfg.get('subset')

                config_dict = {i: cfg[i] for i in cfg.keys()}
                config_dict['dataset_name'] = dataset_name_orig

                dataset_config = DatasetConfig(dataset_name,
                                               child_split,
                                               dataset_class,
                                               root,
                                               caption_path,
                                               features_path,
                                               subset,
                                               config_dict)

                configs.append(dataset_config)
            else:
                print('Invalid dataset {:s} specified'.format(dataset))
                sys.exit(1)

        self.print_info(configs)

        return configs

    def _combine_cfg(self, dataset):
        """If dataset name is separated by 'parent_dataset:child_split' (i.e. 'coco:train2014')
        fallback to parent settings when child configuration has no corresponding parameter
        included"""
        if ":" not in dataset:
            return self.config[dataset], None

        h = dataset.split(':')
        for i in range(-1, -len(h), -1):
            a = ':'.join(h[0:i])
            # print(a)

            # Take defaults from parent and override them as needed:
            for key in self.config[a]:
                if self.config[dataset].get(key) is None:
                    self.config[dataset][key] = self.config[a][key]
        return self.config[dataset], h[-1]

    def _cfg_path(self, cfg, s):
        path = cfg.get(s)
        if path is None or os.path.isabs(path):
            return path
        else:
            root_dir = cfg.get('root_dir', '')
            return os.path.join(root_dir, path)

    def _get_param(self, d, param, default):
        if not d or param not in d or not d[param]:
            return default
        return d[param]

    def print_info(self, configs):
        """Print out details about datasets being configured"""
        for ds in configs:
            print('[Dataset]', ds.name)
            for name, value in ds._asdict().items():
                if name!='name' and name!='config_dict' and value is not None:
                    print('    {}: {}'.format(name, value))
                elif name=='config_dict':
                    print('    {}:'.format(name))
                    for n, v in value.items():
                        print('        {}: {}'.format(n, v))


class ExternalFeature:
    def __init__(self, filename, base_path):
        if base_path is None:
            base_path = ''
        full_path = os.path.expanduser(os.path.join(base_path, filename))
        self.lmdb = None
        self.lmdb_path = None
        self.bin = None
        self.disable_cache = False

        if not os.path.exists(full_path):
            raise FileNotFoundError('ERROR: external feature file not found: ' + full_path)
        if filename.endswith('.h5'):
            import h5py
            self.f = h5py.File(full_path, 'r')
            self.data = self.f['data']
        elif filename.endswith('.lmdb'):
            import lmdb
            self.lmdb = lmdb
            self.lmdb_path = full_path

        elif filename.endswith('.bin'):
            from picsom.bin_data import picsom_bin_data
            self.bin = [ picsom_bin_data(full_path) ]
            self.binx = {}
            print(('PicSOM binary data {:s} contains {:d}' +
                   ' objects of dimensionality {:d}').format(self.bin[0].path(),
                                                             self.bin[0].nobjects(),
                                                             self.bin[0].vdim()))
        elif filename.endswith('.txt'):
            from picsom.bin_data import picsom_bin_data
            self.bin = []
            self.binx = {}
            m = re.match('^(.*/)?[^/]+', full_path)
            assert m
            with open(full_path) as f:
                for p in f:
                    q = p.strip()
                    if q!='' and q[0]!='#':
                        xp = m.group(1)+q
                        i = len(self.bin)
                        self.bin += [ picsom_bin_data(xp) ]
                        print(('PicSOM binary data {:s} contains {:d}' +
                               ' objects of dimensionality {:d}').
                              format(self.bin[i].path(), self.bin[i].nobjects(),
                                     self.bin[i].vdim()))
        else:
            self.data = np.load(full_path)

        x1 = None
        if self.lmdb is not None:
            # Figure out the dimensions of our features:
            with self.lmdb.open(self.lmdb_path, max_readers=1, readonly=True, lock=False,
                                readahead=False, meminit=False) as env:
                with env.begin(write=False) as txn:
                    c = txn.cursor()
                    assert c.first(), full_path
                    x1 = self._lmdb_to_numpy(c.value())

                    # Get feature dimension metadata if available:
                    vdim_data = txn.get('@vdim'.encode('ascii'))
                    if vdim_data is not None:
                        self._vdim = self._lmdb_to_numpy(vdim_data, dtype=np.int32).tolist()
                    else:
                        self._vdim = x1.shape[0]

            # Figure out if our features are too large to cache:
            if np.prod(self._vdim) > 1e5:
                # Subsequent feature loading requests will reinitialize LMDB handle to
                # clear any caches:
                self.disable_cache = True
            else:
                # Caching is on, so we can use the globally defined lmdb handle per featureset:
                self.f = lmdb.open(full_path, max_readers=1, readonly=True, lock=False,
                                   readahead=False, meminit=False)
                self.lmdb = self.f.begin(write=False)
        elif self.bin is not None:
            self._vdim = sum([i.vdim() for i in self.bin])
        else:
            x1 = self.data[0]
            self._vdim = self.data.shape[1]

        assert x1 is None or not np.isnan(x1).any(), full_path

        print('Loaded feature {} with dim {}'.format(full_path, self.vdim()))

    def vdim(self):
        return self._vdim

    def _lmdb_to_numpy(self, value, dtype=np.float32):
        return np.frombuffer(value, dtype=dtype)

    def get_feature(self, idx):
        if self.lmdb is not None:
            if self.disable_cache:
                with self.lmdb.open(self.lmdb_path, max_readers=1, readonly=True, lock=False,
                                    readahead=False, meminit=False) as env:
                    with env.begin(write=False) as txn:
                        x = self._lmdb_to_numpy(txn.get(str(idx).encode('ascii')))
                        x.reshape(self._vdim)
            else:
                try:
                    x = self._lmdb_to_numpy(self.lmdb.get(str(idx).encode('ascii')))
                except:
                    print('No feature data was found with key <{}>'.format(str(idx)))
                    exit(1)
                x.reshape(self._vdim)
        elif self.bin is not None:
            from picsom.bin_data import picsom_bin_data
            pid = os.getpid();
            #print('get_feature()', pid)
            if not pid in self.binx:
                self.binx[pid] = []
                for i in self.bin:
                    self.binx[pid].append(picsom_bin_data(i.path()))
            x = []
            for i in self.binx[pid]:
                found = False
                for j in idx:
                    v = i.get_float(j)
                    if not np.isnan(v[0]):
                        found = True
                        x += v
                        break
                if not found:
                    print('ERROR 1', idx)
                    exit(1)
            #if np.isnan(x).any():
            #    print('ERROR 2', idx, ':', x)
            #print('QQQ', self.bin.path(), idx, x[0:5])
            #assert not np.isnan(x).any(), self.bin.path()+' '+str(idx)
        else:
            x = self.data[idx]

        return torch.tensor(x).float()

    @classmethod
    def load_set(cls, feature_loaders, idx):
        if len(feature_loaders) == 0:
            return None
        else:
            return torch.cat([ef.get_feature(idx) for ef in feature_loaders])

    @classmethod
    def load_sets(cls, feature_loader_sets, idx):
        # We have several sets of features (e.g., initial, persistent, ...)
        # For each set we prepare a single tensor with all the features concatenated
        if feature_loader_sets is None:
            return None
        # NOTE: Removed "if fls" from the list comprehension handling below
        # so that we if we don't specify the initial external features
        # we can still use persistant external features:
        return [cls.load_set(fls, idx) for fls in feature_loader_sets]

    @classmethod
    def loaders(cls, features, base_path):
        ef_loaders = []
        feat_dim = 0

        for fn in features:
            ef = cls(fn, base_path)
            ef_loaders.append(ef)
            # When the number of features is 1 sometimes they come in none 1-d form
            # if statement below takes care of this:
            if len(features) == 1:
                feat_dim = ef.vdim()
                break
            feat_dim += ef.vdim()
        return (ef_loaders, feat_dim)


def tokenize_caption(text, vocab, no_tokenize=False, show_tokens=False,
                     skip_start_token=False):
    """Tokenize a single sentence / caption, convert tokens to vocabulary indices,
    and store the vocabulary index array into a torch tensor"""

    if vocab is None:
        return text

    if no_tokenize:
        tokens = str(text).split()
    else:
        tokens = nltk.tokenize.word_tokenize(str(text).lower())

    caption = []
    if not skip_start_token:
        caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = torch.Tensor(caption)

    if show_tokens:
        joined = ' '.join([vocab(i) for i in caption])
        print('TOKENS', joined)

    return target


def tokenize_paragraph_caption(caption, vocab):
    target = []
    # Get rid of the trailing period before splitting:
    for sentence in caption.rstrip('.').split('.'):
        target.append(tokenize_caption(sentence.strip(), vocab))

    return target


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 iter_over_images=False, feature_loaders=None, config_dict=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json_file: coco annotation file path.
            vocab: vocabulary wrapper.
            subset: file defining a further subset of the dataset to be used
            transform: image transformer.
        """
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(json_file)
        self.iter_over_images = iter_over_images
        if iter_over_images:
            self.ids = list(self.coco.imgs.keys())
        else:
            self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.skip_images = skip_images
        self.feature_loaders = feature_loaders
        self.config_dict = config_dict

        # We are in training mode if vocab is not None, otherwise we might
        # be in vocabulary generation mode, where we don't need to care about
        # the hierarchical model:
        if vocab is not None:
            self.hierarchical_model = config_dict.get('hierarchical_model', False)
            self.max_sentences = config_dict.get('max_sentences')
        else:
            self.hierarchical_model = False

        print("COCO info loaded for {} images and {} captions.".format(len(self.coco.imgs),
                                                                       len(self.coco.anns)))

    def __getitem__(self, index):
        """Returns one training sample as a tuple (image, caption, image_id)."""
        if self.iter_over_images:
            img_id = self.ids[index]
            caption = [a['caption'] for a in self.coco.imgToAnns[img_id]]
            assert self.vocab is None, 'iter_over_images=True and tokenization not supported!'
        else:
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']

        # Get image path
        path = self.coco.loadImgs(img_id)[0]['file_name']
        if not path.startswith('COCO'):  # Yes, this works... for now
            path = 'COCO_val2014_' + path

        if not self.skip_images:
            image = Image.open(os.path.join(self.root, path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = torch.zeros(1, 1)

        # Prepare external features, we use paths to access features
        # NOTE: this only works with lmdb
        feature_sets = ExternalFeature.load_sets(self.feature_loaders, path)

        # See whether we need to prepend start token to the sequence or not:
        skip_start_token = self.config_dict.get('skip_start_token')

        if self.hierarchical_model:
            target = tokenize_paragraph_caption(caption, self.vocab)
        else:
            target = tokenize_caption(caption, self.vocab,
                                      skip_start_token=skip_start_token)

        # We are in feature extraction-only mode,
        # use image filename as image identifier in lmdb:
        if self.vocab is None and self.feature_loaders is None:
            img_id = path

        # We are in file list generation mode and want to output full paths to images:
        if self.config_dict.get('return_full_image_path'):
            img_id = os.path.join(self.root, path)
        # Sometimes we may want just the image file name without full path:
        elif self.config_dict.get('return_image_file_name'):
            img_id = path

        return image, target, img_id, feature_sets

    def __len__(self):
        return len(self.ids)


class VisualGenomeIM2PDataset(data.Dataset):
    """Visual Genome / MS COCO Paragraph-length caption dataset"""

    # FIXME: skip_images, feature_loaders not implemented
    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 iter_over_images=False, feature_loaders=None, config_dict=None):
        """Set the path for images, captions and vocabulary wrapper.
        Args:
            root: image directory.
            json_file: coco annotation file path.
            vocab: vocabulary wrapper.
            subset: file defining a further subset of the dataset to be used
            transform: image transformer.
        """
        self.root = root
        self.vocab = vocab
        self.transform = transform
        self.skip_images = skip_images
        self.feature_loaders = feature_loaders
        self.config_dict = config_dict

        # We are in training mode if vocab is not None, otherwise we might
        # be in vocabulary generation mode, where we don't need to care about
        # the hierarchical model:
        if vocab is not None:
            self.hierarchical_model = config_dict.get('hierarchical_model', False)
            self.max_sentences = config_dict.get('max_sentences')
        else:
            self.hierarchical_model = False

        self.paragraphs = []

        print("Loading Visual Genome Paragraph captioning data from {} ...".
              format(json_file))

        with open(json_file) as data_raw:
            dataset = json.load(data_raw)

        # Assuming subset file is a json array, as provided in
        # https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html
        # If subset is defined, load only selected image/paragraph pairs:
        if subset:
            print("Loading data subset from {} ...".format(subset))
            with open(subset) as subset_raw:
                subset_ids = json.load(subset_raw)
            for d in dataset:
                for img_id in subset_ids:
                    if img_id == d['image_id']:
                        self.paragraphs.append({
                            'image_id': img_id,
                            'caption': d['paragraph']
                        })
                        break
        # Otherwise load all images in json_file:
        else:
            print("Loading all data...")
            for d in dataset:
                self.paragraphs.append({
                    'image_id': d['image_id'],
                    'caption': d['paragraph']
                })

        print("VisualGenome paragraph data loaded for {} images...".format(
            len(self.paragraphs)))

    def __getitem__(self, index):
        """Returns one data pair (image and paragraph)."""
        caption = self.paragraphs[index]['caption']
        img_id = self.paragraphs[index]['image_id']
        img_filename = str(img_id) + '.jpg'
        path = os.path.join(self.root, img_filename)

        if not self.skip_images:
            image = Image.open(path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = torch.zeros(1, 1)

        # Prepare external features
        # TODO probably wrong index ...
        feature_sets = ExternalFeature.load_sets(self.feature_loaders, img_filename)

        # For hierarchical model the sentences need to be separated:
        if self.hierarchical_model:
            target = tokenize_paragraph_caption(caption, self.vocab)
        # Otherwise treat each paragraph as a single sentence:
        else:
            target = tokenize_caption(caption, self.vocab)

        # We are in feature extraction-only mode,
        # use image filename as image identifier in lmdb:
        if self.vocab is None and self.feature_loaders is None:
            img_id = path

        # We are in file list creation mode, extract full path:
        if self.config_dict.get('return_full_image_path'):
            img_id = path

        # Sometimes we may want just the image file name without full path:
        elif self.config_dict.get('return_image_file_name'):
            img_id = img_filename

        return image, target, img_id, feature_sets

    def __len__(self):
        return len(self.paragraphs)


class VistDataset(data.Dataset):
    """VIST Custom Dataset for sequence processing, compatible with torch.utils.data.DataLoader."""

    # FIXME: skip_images, feature_loaders not implemented
    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 iter_over_images=False, feature_loaders=None, config_dict=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json_file: VIST annotation file path.
            vocab: vocabulary wrapper.
            subset: file defining a further subset of the dataset to be used
            transform: image transformer.
        """
        self.root = root
        self.vocab = vocab
        self.transform = transform

        # Get the list of available images:
        images = [str(file).split('.')[0] for file in os.listdir(root)]

        with open(json_file) as raw_data:
            json_data = json.load(raw_data)
            self.anns = json_data['annotations']

        seq_idx = 0
        self.data_hold = []
        self.story_ids = []
        while seq_idx < len(self.anns):
            current_story_id = self.anns[seq_idx][0]['story_id']
            _seq_idx = seq_idx
            current_story = str()
            current_sequence = []
            bad_for_testing = False
            while _seq_idx < len(self.anns) and self.anns[_seq_idx][0]['story_id'] == current_story_id:
                current_story += self.anns[_seq_idx][0]['text']
                current_sequence.append(self.anns[_seq_idx][0]['photo_flickr_id'])

                # local testing purpose (needs to be removed)
                if self.anns[_seq_idx][0]['photo_flickr_id'] not in images:
                    bad_for_testing = True

                _seq_idx += 1

            seq_idx = _seq_idx
            # local testing purpose (needs to be removed)
            if not bad_for_testing:
                self.data_hold.append([current_sequence, current_story])
                self.story_ids.append(current_story_id)

        print("... {} sequences loaded ...".format(len(self.story_ids)))

    def __getitem__(self, index):
        """Returns one training sample as a tuple (image, caption, image_id)."""

        vocab = self.vocab
        image_ids = self.data_hold[index][0]
        story = self.data_hold[index][1]

        sequence = []
        for image_id in image_ids:
            image_path = os.path.join(self.root, str(image_id) + '.jpg')
            if os.path.isfile(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                image_path = os.path.join(self.root, str(image_id) + '.png')
                image = Image.open(image_path).convert('RGB')

            if self.transform is not None:
                image = self.transform(image)

            sequence.append(image)

        # Convert caption (string) to word ids.
        target = tokenize_caption(story, vocab)

        return sequence, target, self.story_ids[index]

    def __len__(self):
        return len(self.data_hold)


class MSRVTTDataset(data.Dataset):
    """MSR-VTT Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 iter_over_images=False, feature_loaders=None, config_dict=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json_file: path to train_val_videodatainfo.json.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.vocab = vocab
        self.transform = transform
        self.skip_images = skip_images
        self.feature_loaders = feature_loaders
        self.subset = subset if subset else 'train'

        self.captions = []
        subset_vids = set()

        with open(json_file, 'r') as fp:
            j = json.load(fp)
            for v in j['videos']:
                if v['split'] == self.subset:
                    subset_vids.add(v['video_id'])

            for s in j['sentences']:
                vid = s['video_id']
                if vid in subset_vids:
                    self.captions.append((vid, s['caption']))

        print("MSR-VTT info [{}] loaded for {} images, {} captions.".format(self.subset,
                                                                            len(subset_vids), len(self.captions)))

    def __getitem__(self, index):
        """Returns one training sample as a tuple (image, caption, image_id)."""

        vid = self.captions[index][0]
        caption = self.captions[index][1]

        assert vid[:5] == 'video'
        vid_idx = int(vid[5:])
        path = '{:04}:kf1.jpeg'.format(vid_idx)

        if not self.skip_images:
            image = Image.open(os.path.join(self.root, path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = torch.zeros(1, 1)

        # Prepare external features
        feature_sets = ExternalFeature.load_sets(self.feature_loaders, vid_idx)

        # Convert caption (string) to word ids.
        vocab = self.vocab
        target = tokenize_caption(caption, vocab)

        return image, target, vid_idx, feature_sets

    def __len__(self):
        return len(self.captions)


class TRECVID2018Dataset(data.Dataset):
    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 iter_over_images=False, feature_loaders=None, config_dict=None):
        self.root = root
        self.vocab = vocab
        self.transform = transform
        self.skip_images = skip_images
        self.feature_loaders = feature_loaders

        self.id_to_filename = {}
        for filename in glob.glob(self.root + '/*.jpeg'):
            m = re.match(r'(\d+):\d+$', basename(filename))
            if m:
                image_id = int(m.group(1))
                assert image_id >= 0, 'filename={}'.format(filename)
                assert image_id not in self.id_to_filename
                self.id_to_filename[image_id] = filename
            else:
                print('WARNING: filename {} could not be parsed, skipping...'.format(filename))

        print("TRECVID 2018 info loaded for {} images.".format(len(self)))

    def __getitem__(self, index):
        """Returns one training sample as a tuple (image, caption, image_id)."""

        filename = self.id_to_filename[index]

        if not self.skip_images:
            image_path = os.path.join(self.root, filename)
            image = Image.open(image_path)
            image = image.resize([224, 224], Image.LANCZOS)
            if image.mode != 'RGB':
                print('WARNING: converting {} from {} to RGB'.
                      format(image_path, image.mode))
                image = image.convert('RGB')

            if self.transform is not None:
                image = self.transform(image)  # .unsqueeze(0)
        else:
            image = torch.zeros(1, 1)

        # Prepare external features
        feature_sets = ExternalFeature.load_sets(self.feature_loaders, index)

        return image, None, index, feature_sets

    def __len__(self):
        return len(self.id_to_filename)


class PicSOMDataset(data.Dataset):
    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 iter_over_images=False, feature_loaders=None, config_dict=None):
        from picsom.label_index import picsom_label_index
        from picsom.class_file  import picsom_class
        from picsom.bin_data    import picsom_bin_data
    
        self.root = root
        self.vocab = vocab
        self.subset = subset
        self.transform = transform
        self.skip_images = skip_images
        self.feature_loaders = feature_loaders
        self.picsom_root      = config_dict['picsom_root']
        self.picsom_database  = config_dict['dataset_name'].split(':')[1]
        self.picsom_label_map = config_dict.get('label_map', None)
        self.no_tokenize      = config_dict.get('no_tokenize', False)
        self.show_tokens      = config_dict.get('show_tokens', False)

        print('PicSOM root={:s} database={:s}'.format(self.picsom_root,
                                                      self.picsom_database))
        
        self.db_root = self.picsom_root+'/databases/'+self.picsom_database

        self.labels = picsom_label_index(self.db_root+'/labels.txt')
        print('PicSOM database {:s} contains {:d} objects'.format(self.picsom_database, 
                                                                  self.labels.nobjects()))

        self.use_lmdb = self.feature_loaders is not None and \
                        len(self.feature_loaders[0]) and \
                        self.feature_loaders[0][0].lmdb is not None
        print('PicSOM using {} features'.format('LMDB' if self.use_lmdb else 'BIN'))

        subset = self.db_root+'/classes/'+self.subset
        # print(subset)
        assert os.path.isfile(subset), 'subset class file <'+subset+'> inexistent'
        self.restr = picsom_class(subset)
        restr_size = len(self.restr.objects())
        print('PicSOM class file {:s} contains {:d} objects'.format(self.restr.path(),
                                                                    restr_size))

        restr_set = self.restr.objects()
        
        label_map = {}
        if self.picsom_label_map is not None:
            ll = {}
            tt = self.db_root+"/"+self.picsom_label_map
            with open(tt) as fp:
                for l in fp:
                    l = l.rstrip()
                    #print(l)
                    a = re.match('([^ ]+) (.*)', l)
                    assert a, 'reading label_map failed'
                    label = a.group(1)
                    if label in restr_set:
                        ll[label] = 1
                        label_map[label] = a.group(2)
            print('PicSOM {} label map entries loaded from {}'.
                  format(len(ll), self.picsom_label_map))

        self.data = []
        tt = json_file
        ll = {}
        # print(tt)
        with open(tt) as fp:
            for l in fp:
                l = l.rstrip()
                #print('<{}>'.format(l))
                a = re.match('([^ ]+)( (.*))?', l)
                assert a, 'reading <'+tt+'> failed'
                label = a.group(1)
                if label in restr_set:
                    ll[label] = 1
                    idxs = []
                    if not self.use_lmdb:
                        idxs.append(self.labels.index_by_label(label))
                        laa = label_map.get(label, '').split()
                        for la in laa:
                            idxs.append(self.labels.index_by_label(la))
                            
                    texts = a.group(3).split(' # ')
                    #if len(texts)!=1:
                    #    print(label, len(texts))
                    for text in texts:
                        self.data.append((label, text, idxs))
        
        print('PicSOM {} texts loaded for {} images from {}'.
              format(len(self.data), len(ll), tt))
        

    def __getitem__(self, index):
        """Returns one training sample as a tuple (image, caption, image_id)."""

        label_text_index = self.data[index]
        label = label_text_index[0]
        label_or_idx = label if self.use_lmdb else label_text_index[2]
        if False:
            print('PicSOMDataset.getitem() {:d} {:s} {!s:s}'.
                  format(index, label, label_or_idx))

        target  = tokenize_caption(label_text_index[1], self.vocab,
                                   no_tokenize=self.no_tokenize, show_tokens=self.show_tokens)
    
        feature_sets = ExternalFeature.load_sets(self.feature_loaders, label_or_idx)

        #print('__getitem__ ending', target, feature_sets)
        
        return torch.zeros(1, 1), target, label, feature_sets

    def __len__(self):
        return len(self.data)
        

class GenericDataset(data.Dataset):
    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 iter_over_images=False, feature_loaders=None, config_dict=None):
        self.vocab = vocab
        self.transform = transform
        self.skip_images = skip_images
        self.feature_loaders = feature_loaders

        if type(root) is list:
            self.filelist = root
        elif os.path.isdir(root):
            self.filelist = []
            for filename in glob.glob(root + '/*.jpeg'):
                self.filelist.append(filename)
        else:
            print('ERROR: root neither file list or dir!')
            sys.exit(1)

        print("GenericDataset: loaded {} images.".format(len(self.filelist)))

    def __getitem__(self, index):
        """Returns one training sample as a tuple (image, caption, image_id)."""

        image_path = self.filelist[index]

        if not self.skip_images:
            image = Image.open(image_path)
            image = image.resize([224, 224], Image.LANCZOS)
            if image.mode != 'RGB':
                print('WARNING: converting {} from {} to RGB'.
                      format(image_path, image.mode))
                image = image.convert('RGB')

            if self.transform is not None:
                image = self.transform(image)
        else:
            image = torch.zeros(1, 1)

        # Prepare external features
        path = os.path.splitext(os.path.basename(image_path))[0]
        feature_sets = ExternalFeature.load_sets(self.feature_loaders, path)

        return image, None, path, feature_sets

    def __len__(self):
        return len(self.filelist)


def _collate_features(feature_sets, sorting_idx=None):
    # Generate list of (batch_size, concatenated_feature_length) tensors,
    # one for each feature set
    # feature_sets is a tuple of lists -
    # -- one tuple corresponds to an image
    # -- list elements correspond to different feature types for the same image
    # :param sorting_idx: used by hierarchical model to sort the features 
    #                     to match the initial sorting order when collating
    if feature_sets[0] is not None:
        batch_size = len(feature_sets)
        num_feature_sets = len(feature_sets[0])
        features = list()

        for fs_i in range(num_feature_sets):
            if feature_sets[0][fs_i] is not None:
                current_feature = torch.stack([feature_sets[i][fs_i] for i in range(batch_size)], 0)
                if sorting_idx is not None:
                    assert len(sorting_idx) == len(current_feature)
                    current_feature = current_feature[sorting_idx]
                features.append(current_feature)
            else:
                # Allow None features so that the feature type index in train.py stays correct
                features.append(None)
    else:
        features = None

    return features


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, image_ids).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    # If we are doing inference, captions are None, and we skip the sort
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, indices, feature_sets = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    if type(captions[0]) is str:  # we are returning non-tokenized strings
        targets = captions
        lengths = None
    elif type(captions[0]) is list:  # we are returning a list of strings
        targets = captions
        lengths = None
    elif captions[0] is not None:
        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
    else:
        # If we are doing inference, captions are None...
        lengths = None
        targets = None

    features = _collate_features(feature_sets)

    return images, targets, lengths, indices, features


def collate_fn_vist(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, story_ids = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    # images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths, story_ids


def collate_hierarchical(data, max_sentences, vocab):
    """Outputs data in a format compatible with hierarchical Decoder

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
            - img_id: id of the image in the dataset
            - feature_set

    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each sentence in a paragraph.
        image_ids:
        features: a list of size 2 (0 - initial features, 1 - persistant features)
        sorting_order: sorting order relative to the first sentence of a paragraph
                       needed to do pack-padded sequence based operations on individual
                       sentences
        last_sentence_indicator: tensor of size (batch_size x max_sentences) storing "1"
                                 at positions indicating last sentence in the paragraph
    """

    images, captions, image_ids, feature_sets = zip(*data)

    batch_size = len(captions)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Get the length of the longest sentence:
    _lengths = [[len(sentence) for sentence in paragraph] for paragraph in captions]

    max_length = max(max(_lengths, key=lambda x: max(x)))

    # Placeholder tensor for all captions, of dimension:
    #   (batchsize X max allowed sentences per paragraph X longest sentence)
    targets = torch.zeros(batch_size, max_sentences, max_length).long()

    # Placeholder tensor for all sentence lengths, of dimension
    #   (batchsize X max allowed sentences per paragraph)
    lengths = torch.zeros(batch_size, max_sentences).long()

    last_sentence_indicator = torch.zeros(batch_size, max_sentences).long()

    # Generate a table storing data about which sentence in a
    # paragraph is the last one:
    for i, paragraph in enumerate(captions):
        s = len(paragraph)
        if s < max_sentences:
            last_sentence_indicator[i, s - 1] = 1
        else:
            last_sentence_indicator[i, max_sentences - 1] = 1

    # Placeholder tensor for sorted image ids, of dimensions
    #   (max allowed sentences per paragraph X batchsize)
    # (NOTE: The sorting order for the sentence RNN is the same as the
    # sorting order for the first word RNN)
    # This shouldn't be a tensor! IDs can be anything!
    # image_ids_sorted = torch.zeros(len(captions), max_sentences)

    for i in range(batch_size):
        # This step also limits the number of sentences per paragraph
        # to be <= max_captions
        for j in range(max_sentences):
            if len(captions[i]) > j:
                lengths[i, j] = len(captions[i][j])
                targets[i, j, :lengths[i, j]] = captions[i][j]

    ##################################################################
    # Sort everything based on the length of first sentence of each  #
    # paragraph caption in the minibatch:                            #
    ##################################################################
    # sorting_order = [[] for _ in range(max_sentences)]
    sorting_order = torch.zeros(batch_size, max_sentences).long()
    # First create a default sorting order:
    # (Good to know: torch.sort() outputs a tuple of sorted data structure and sorted indices)
    _, idxs_sorted_0 = torch.sort(lengths[:, 0], descending=True)

    lengths = lengths[idxs_sorted_0]
    targets = targets[idxs_sorted_0]
    # Sort image ids:
    image_ids = [image_ids[i] for i in idxs_sorted_0]
    last_sentence_indicator = last_sentence_indicator[idxs_sorted_0]
    # Sort images also (but only once), images are sorted in the
    # same order as inputs for the first WordRNN:
    images = images[idxs_sorted_0]

    ##############################################################
    # Store sorting order indexed by sentence in the mini batch: #
    ##############################################################
    # For the first word-rnn we keep the same sorting order as in the SentenceRNN:
    sorting_order[:, 0] = torch.tensor([i for i in range(len(idxs_sorted_0))])

    for j in range(1, max_sentences):
        lengths[:, j], idxs_sorted = torch.sort(lengths[:, j], descending=True)
        targets[:, j] = targets[:, j][idxs_sorted]
        sorting_order[:, j] = idxs_sorted

    features = _collate_features(feature_sets, idxs_sorted_0)

    return (images, targets, lengths, image_ids, features,
            sorting_order, last_sentence_indicator)


def unzip_image_dir(image_dir):
    # Check if $TMPDIR envirnoment variable is set and use that
    env_tmp = os.environ.get('TMPDIR')
    # Also check if the environment variable points to '/tmp/some/dir' to avoid
    # nasty surprises
    if env_tmp and os.path.commonprefix([os.path.abspath(env_tmp), '/tmp']) == '/tmp':
        tmp_root = os.path.abspath(env_tmp)
    else:
        tmp_root = '/tmp'

    extract_path = os.path.join(tmp_root)

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    with zipfile.ZipFile(image_dir, 'r') as zipped_images:
        print("Extracting training data from {} to {}".format(image_dir, extract_path))
        zipped_images.extractall(extract_path)
        unzipped_dir = os.path.basename(image_dir).split('.')[0]
        return os.path.join(extract_path, unzipped_dir)


def get_dataset_class(cls_name):
    """Return the correct dataset class based on the one specified in configuration file"""
    if cls_name == 'CocoDataset':
        return CocoDataset
    elif cls_name == 'VisualGenomeIM2PDataset':
        return VisualGenomeIM2PDataset
    elif cls_name == 'VistDataset':
        return VistDataset
    elif cls_name == 'MSRVTTDataset':
        return MSRVTTDataset
    elif cls_name == 'TRECVID2018Dataset':
        return TRECVID2018Dataset
    elif cls_name == 'PicSOMDataset':
        return PicSOMDataset
    elif cls_name == 'GenericDataset':
        return GenericDataset
    else:
        print('Invalid dataset {:s} specified'.format(cls_name))
        sys.exit(1)


def set_collate_arguments(func, **kwargs):
    """Outputs a collate function with all configurable arguments set
    this is needed for data_loader constructur which expects the collate function
    to be run with one argument only - data, so we need the following wrapper
    to specify other needed parameters"""
    def f(data):
        return func(data, **kwargs)
    return f


def get_loader(dataset_configs, vocab, transform, batch_size, shuffle, num_workers,
               ext_feature_sets=None, skip_images=False, iter_over_images=False,
               _collate_fn=collate_fn, verbose=False):
    """Returns torch.utils.data.DataLoader for user-specified dataset."""

    datasets = []

    for dataset_config in dataset_configs:
        dataset_cls = get_dataset_class(dataset_config.dataset_class)
        root = dataset_config.image_dir
        json_file = dataset_config.caption_path
        subset = dataset_config.subset
        fpath = dataset_config.features_path
        config_dict = dataset_config.config_dict

        loaders = None
        dims = None
        if ext_feature_sets is not None:
            # Construct external feature loaders for each of the specified feature sets
            loaders_and_dims = [ExternalFeature.loaders(fs, fpath) for fs in ext_feature_sets]

            # List of tuples into two lists...
            loaders, dims = zip(*loaders_and_dims)

        # Unzip training images to /tmp/data if image_dir argument points to zip file:
        if isinstance(root, str) and zipfile.is_zipfile(root):
            root = unzip_image_dir(root)

        if verbose:
            print((' root={!s:s}\n json_file={!s:s}\n vocab={!s:s}\n subset={!s:s}\n'+
                   ' transform={!s:s}\n skip_images={!s:s}\n iter_over_images={!s:s}\n'+
                   ' loaders={!s:s}\n config_dict={!s:s}').format(root, json_file, vocab,
                                                                  subset, transform,
                                                                  skip_images, iter_over_images,
                                                                  loaders, config_dict))

        dataset = dataset_cls(root=root, json_file=json_file, vocab=vocab,
                              subset=subset, transform=transform, skip_images=skip_images,
                              iter_over_images=iter_over_images, feature_loaders=loaders,
                              config_dict=config_dict)

        datasets.append(dataset)

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = data.ConcatDataset(datasets)

    # Hierarchical model requires a different collate function,
    # also when vocab is set to None it means that we are building the vocabulary,
    # and thus should not be honoring the hierarchical_model specific stuff when loading 
    # the model
    if config_dict.get('hierarchical_model') and vocab is not None:
        print('Preparing collate callable for hierarchical model..')
        max_sentences = config_dict['max_sentences']
        _collate_fn = set_collate_arguments(collate_hierarchical,
                                            max_sentences=max_sentences, vocab=vocab)
    else:
        _collate_fn = _collate_fn

    # Data loader:
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption.
    # length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=_collate_fn)
    return data_loader, dims


if __name__ == '__main__':
    # Test loading dataset!
    from pprint import pprint

    vg_root = 'datasets/data/VisualGenome'
    vocab_path = 'datasets/processed/COCO/vocab.pkl'
    with open(vocab_path, 'rb') as f:
        print("Extracting vocabulary from {}".format(vocab_path))
        vocab = pickle.load(f)

    # Load VG Paragraph training subset:
    vgim2p_subset = VisualGenomeIM2PDataset(root=vg_root + '/1.2/VG/1.2/images',
                                            json_file=vg_root + '/im2p/paragraphs_v1.json',
                                            vocab=vocab,
                                            subset=vg_root + '/im2p/train_split.json')
    pprint(vgim2p_subset[0])
    pprint(vgim2p_subset[10000])
    pprint(vgim2p_subset[-1])
    # Load VG Paragraph full dataset:
    vgim2p_full = VisualGenomeIM2PDataset(root=vg_root + '/1.2/VG/1.2/images',
                                          json_file=vg_root + '/im2p/paragraphs_v1.json',
                                          vocab=vocab)
    pprint(vgim2p_full[0])
    pprint(vgim2p_full[10000])
    pprint(vgim2p_full[-1])
