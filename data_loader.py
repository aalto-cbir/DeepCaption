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

from vocabulary import Vocabulary  # (Needed to handle Vocabulary pickle)
from collections import namedtuple
from PIL import Image
import configparser


def basename(fname):
    return os.path.splitext(os.path.basename(fname))[0]


DatasetConfig = namedtuple('DatasetConfig',
                           'name, child_split, dataset_class, image_dir, caption_path, '
                           'vocab_path, features_path, subset, config_dict')


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
            print('Config file not found. Loading default settings for generic dataset.')
            print('Hint: you can use datasets/datasets.conf.default as a starting point.')

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

    def get_params(self, args_dataset, image_dir=None, image_files=None, vocab_path=None):

        datasets = args_dataset.split('+')
        configs = []
        for dataset in datasets:
            dataset_name_orig = dataset # needed if lower-casing is used
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
                cfg_vocab_path = vocab_path if vocab_path else self._cfg_path(cfg, 'vocab_path')
                features_path = self._cfg_path(cfg, 'features_path')
                subset = cfg.get('subset')

                config_dict = { i: cfg[i] for i in cfg.keys() }
                config_dict['dataset_name'] = dataset_name_orig
                
                dataset_config = DatasetConfig(dataset_name,
                                               child_split,
                                               dataset_class,
                                               root,
                                               caption_path,
                                               cfg_vocab_path,
                                               features_path,
                                               subset,
                                               config_dict)

                configs.append(dataset_config)
            else:
                print('Invalid dataset {:s} specified'.format(dataset))
                sys.exit(1)

        # Vocab path can be overriden from arguments even for multiple datasets:
        main_vocab_path = vocab_path if vocab_path else self._cfg_path(
            self.config[datasets[0]], 'vocab_path')

        if main_vocab_path is None:
            print("WARNING: Vocabulary path not specified!")

        self.print_info(configs)

        return configs, main_vocab_path

    def _combine_cfg(self, dataset):
        """If dataset name is separated by 'parent_dataset:child_split' (i.e. 'coco:train2014')
        fallback to parent settings when child configuration has no corresponding parameter
        included"""
        if ":" not in dataset:
            return self.config[dataset], None

        h = dataset.split(':')
        for i in range(-1, -len(h), -1):
            a = ':'.join(h[0:i])
            print(a)

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
                if name != 'name' and value is not None:
                    print('    {}: {}'.format(name, value))


class ExternalFeature:
    def __init__(self, filename, base_path):
        full_path = os.path.expanduser(os.path.join(base_path, filename))
        self.lmdb = None
        self.bin  = None
        if not os.path.exists(full_path):
            print('ERROR: external feature file not found:', full_path)
            sys.exit(1)
        if filename.endswith('.h5'):
            import h5py
            self.f = h5py.File(full_path, 'r')
            self.data = self.f['data']
        elif filename.endswith('.lmdb'):
            import lmdb
            self.f = lmdb.open(full_path, max_readers=1, readonly=True, lock=False,
                               readahead=False, meminit=False)
            self.lmdb = self.f.begin(write=False)
        elif filename.endswith('.bin'):
            from picsom_bin_data import picsom_bin_data
            self.bin = picsom_bin_data(full_path)
            print(('PicSOM binary data {:s} contains {:d}'+
                   ' objects of dimensionality {:d}').format(self.bin.path(),
                                                             self.bin.nobjects(),
                                                             self.bin.vdim()))
        else:
            self.data = np.load(full_path)

        x1 = None
        if self.lmdb is not None:
            c = self.lmdb.cursor()
            assert c.first(), full_path
            x1 = self._lmdb_to_numpy(c.value())
            self._vdim = x1.shape[0]
        elif self.bin is not None:
            self._vdim = self.bin.vdim()
        else:
            x1 = self.data[0]
            self._vdim = self.data.shape[1]

        assert x1 is None or not np.isnan(x1).any(), full_path

        print('Loaded feature {} with dim {}.'.format(full_path, self.vdim()))

    def vdim(self):
        return self._vdim

    def _lmdb_to_numpy(self, value):
        return np.frombuffer(value, dtype=np.float32)

    def get_feature(self, idx):
        if self.lmdb is not None:
            x = self._lmdb_to_numpy(self.lmdb.get(str(idx).encode('ascii')))
        elif self.bin is not None:
            x = self.bin.get_float(idx)
            #if np.isnan(x).any():
            #    print('ERROR', idx, ':', x)
            #assert not np.isnan(x).any(), self.bin.path()+' '+str(idx)
        else:
            x = self.data[idx]

        return torch.tensor(x).float()

    @classmethod
    def load_set(cls, feature_loaders, idx):
        return torch.cat([ef.get_feature(idx) for ef in feature_loaders])

    @classmethod
    def load_sets(cls, feature_loader_sets, idx):
        # We have several sets of features (e.g., initial, persistent, ...)
        # For each set we prepare a single tensor with all the features concatenated
        if feature_loader_sets is None:
            return None
        return [cls.load_set(fls, idx) for fls in feature_loader_sets
                if fls]

    @classmethod
    def loaders(cls, features, base_path):
        ef_loaders = []
        feat_dim = 0
        for fn in features:
            ef = cls(fn, base_path)
            ef_loaders.append(ef)
            feat_dim += ef.vdim()
        return (ef_loaders, feat_dim)


def tokenize_caption(text, vocab):
    """Tokenize a single sentence / caption, convert tokens to vocabulary indices,
    and store the vocabulary index array into a torch tensor"""

    if vocab is None:
        return text

    tokens = nltk.tokenize.word_tokenize(str(text).lower())
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = torch.Tensor(caption)

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
        target = tokenize_caption(caption, self.vocab)

        # We are in feature extraction-only mode,
        # use image filename as image identifier in lmdb:
        if self.vocab is None and self.feature_loaders is None:
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

        print("VisualGenome paragraph data loaded for {} images...".format(len(self.paragraphs)))

    def __getitem__(self, index):
        """Returns one data pair (image and paragraph)."""
        cap = self.paragraphs[index]['caption']
        img_id = self.paragraphs[index]['image_id']
        path = os.path.join(self.root, str(img_id) + '.jpg')

        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Prepare external features
        # TODO probably wrong index ...
        feature_sets = ExternalFeature.load_sets(self.feature_loaders, path)
        target = tokenize_caption(cap, self.vocab)

        # We are in feature extraction-only mode,
        # use image filename as image identifier in lmdb:
        if self.vocab is None and self.feature_loaders is None:
            img_id = path

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
        from picsom_label_index import picsom_label_index
        from picsom_class       import picsom_class
        from picsom_bin_data    import picsom_bin_data
    
        self.root = root
        self.vocab = vocab
        self.subset = subset
        self.transform = transform
        self.skip_images = skip_images
        self.feature_loaders = feature_loaders
        self.picsom_root = config_dict['picsom_root']
        self.picsom_database = config_dict['dataset_name'].split(':')[1]

        print('PicSOM root={:s} database={:s}'.format(self.picsom_root,
                                                      self.picsom_database))
        
        self.db_root = self.picsom_root+'/databases/'+self.picsom_database

        self.labels = picsom_label_index(self.db_root+'/labels.txt')
        print('PicSOM database {:s} contains {:d} objects'.format(self.picsom_database, 
                                                                  self.labels.nobjects()))

        self.use_lmdb = self.feature_loaders is not None and \
                        self.feature_loaders[0][0].lmdb is not None
        print('PicSOM using {} features'.format('LMDB' if self.use_lmdb else 'BIN'))

        subset = self.db_root+'/classes/'+self.subset
        # print(subset)
        self.restr = picsom_class(subset)
        restr_size = len(self.restr.objects())
        print('PicSOM class file {:s} contains {:d} objects'.format(self.restr.path(),
                                                                    restr_size))
        restr_set = self.restr.objects()
        
        self.texts = []
        tt = json_file
        ll = {}
        # print(tt)
        with open(tt) as fp:
            for l in fp:
                l = l.rstrip()
                #print(l)
                a = re.match('([^ ]+) (.*)', l)
                assert a
                label = a.group(1)
                if label in restr_set:
                    ll[label] = 1
                    texts = a.group(2).split(' # ')
                    #if len(texts)!=1:
                    #    print(label, len(texts))
                    for text in texts:
                        self.texts.append((label, text))
        
        print('PicSOM {} texts loaded for {} images from {}'.
              format(len(self.texts), len(ll), tt))

    def __getitem__(self, index):
        """Returns one training sample as a tuple (image, caption, image_id)."""

        label_text = self.texts[index]
        label = label_text[0]
        bin_data_idx = -1 if self.use_lmdb else self.labels.index_by_label(label)
        # print('PicSOMDataset.getitem() {:d} {:s} {:d}'.format(index, label, bin_data_idx))
        
        image = torch.zeros(1, 1)

        caption = label_text[1]
        target  = tokenize_caption(caption, self.vocab)

        feature_sets = ExternalFeature.load_sets(self.feature_loaders, 
                                                 label if self.use_lmdb else bin_data_idx)

        #print('__getitem__ ending', target, feature_sets)
        
        return image, target, index, feature_sets

    def __len__(self):
        return len(self.texts)
        

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

    # Generate list of (batch_size, concatenated_feature_length) tensors,
    # one for each feature set
    if feature_sets[0] is not None:
        batch_size = len(feature_sets)
        num_feature_sets = len(feature_sets[0])
        features = [torch.stack([feature_sets[i][fs_i] for i in range(batch_size)], 0)
                    for fs_i in range(num_feature_sets)]
    else:
        features = None

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


def get_loader(dataset_configs, vocab, transform, batch_size, shuffle, num_workers,
               ext_feature_sets=None, skip_images=False, iter_over_images=False,
               _collate_fn=collate_fn):
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

        # if verbose:
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
