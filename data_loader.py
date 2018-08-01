import glob
import json
import nltk
import os
import re
import sys

import numpy as np
import torch
import torch.utils.data as data

from PIL import Image


def basename(fname):
    fname.split(':')
    return os.path.splitext(os.path.basename(fname))[0]


class ExternalFeature:
    def __init__(self, filename, base_path):
        full_path = os.path.expanduser(os.path.join(base_path, filename))
        if filename.endswith('.h5'):
            import h5py
            self.f = h5py.File(full_path, 'r')
            self.data = self.f['data']
        else:
            self.data = np.load(full_path)
        x1 = self.data[0]
        assert not np.isnan(x1).any(), full_path

    def vdim(self):
        return self.data.shape[1]

    def get_feature(self, idx):
        return torch.tensor(self.data[idx])

    @classmethod
    def load_set(cls, feature_loaders, idx):
        return torch.cat([ef.get_feature(idx) for ef in feature_loaders])

    @classmethod
    def load_sets(cls, feature_loader_sets, idx):
        # We have several sets of features (e.g., initial, persistent, ...)
        # For each set we prepare a single tensor with all the features concatenated
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


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 feature_loaders=None):
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
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.skip_images = skip_images
        self.feature_loaders = feature_loaders
        print("COCO info loaded for {} images.".format(len(self.ids)))

    def __getitem__(self, index):
        """Returns one training sample as a tuple (image, caption, image_id)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']

        if not self.skip_images:
            path = coco.loadImgs(img_id)[0]['file_name']

            image = Image.open(os.path.join(self.root, path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = None

        # Prepare external features
        # TODO probably wrong index ...
        feature_sets = ExternalFeature.load_sets(self.feature_loaders, index)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, img_id, feature_sets

    def __len__(self):
        return len(self.ids)


class VisualGenomeIM2PDataset(data.Dataset):
    """Visual Genome / MS COCO Paragraph-length caption dataset"""

    # FIXME: skip_images, feature_loaders not implemented
    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 feature_loaders=None):
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

        print("... {} images loaded ...".format(len(self.paragraphs)))

    def __getitem__(self, index):
        """Returns one data pair (image and paragraph)."""
        vocab = self.vocab
        cap = self.paragraphs[index]['caption']
        img_id = self.paragraphs[index]['image_id']
        img_path = os.path.join(self.root, str(img_id) + '.jpg')

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(cap).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.paragraphs)


class VistDataset(data.Dataset):
    """VIST Custom Dataset for sequence processing, compatible with torch.utils.data.DataLoader."""

    # FIXME: skip_images, feature_loaders not implemented
    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 feature_loaders=None):
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
        tokens = nltk.tokenize.word_tokenize(str(story).lower())
        story = [vocab('<start>')]
        story.extend([vocab(token) for token in tokens])
        story.append(vocab('<end>'))
        target = torch.Tensor(story)
        return sequence, target, self.story_ids[index]

    def __len__(self):
        return len(self.data_hold)


class MSRVTTDataset(data.Dataset):
    """MSR-VTT Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 feature_loaders=None):
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

        self.captions = []
        train_vids = set()

        with open(json_file, 'r') as fp:
            j = json.load(fp)
            for v in j['videos']:
                if v['split'] == 'train':
                    train_vids.add(v['video_id'])

            for s in j['sentences']:
                vid = s['video_id']
                if vid in train_vids:
                    self.captions.append((vid, s['caption']))

        print("MSR-VTT info loaded for {} images, {} captions.".format(
            len(train_vids), len(self.captions)))

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
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return image, target, vid_idx, feature_sets

    def __len__(self):
        return len(self.captions)


class TRECVID2018Dataset(data.Dataset):
    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 feature_loaders=None):
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


class GenericDataset(data.Dataset):
    def __init__(self, root, json_file, vocab, subset=None, transform=None, skip_images=False,
                 feature_loaders=None):
        self.filelist = root
        self.vocab = vocab
        self.transform = transform
        self.skip_images = skip_images
        self.feature_loaders = feature_loaders

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
        feature_sets = ExternalFeature.load_sets(self.feature_loaders, index)

        return image, None, image_path, feature_sets

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

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    # If we are doing inference, captions are None...
    if captions[0] is not None:
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
    else:
        lengths = None
        targets = None

    # Generate list of (batch_size, concatenated_feature_length) tensors,
    # one for each feature set
    batch_size = len(feature_sets)
    num_feature_sets = len(feature_sets[0])
    features = [torch.stack([feature_sets[i][fs_i] for i in range(batch_size)], 0)
                for fs_i in range(num_feature_sets)]

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


def get_loader(dataset_name, root, json_file, vocab, transform, batch_size,
               shuffle, num_workers, subset=None, skip_images=False,
               feature_loaders=None, _collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for user-specified dataset."""

    dn = dataset_name.lower()

    if dn == 'coco':
        _dataset = CocoDataset
    elif 'vist' in dn:
        _dataset = VistDataset
    elif dn == 'vgim2p':
        _dataset = VisualGenomeIM2PDataset
    elif dn == 'msrvtt' or dn == 'msr-vtt':
        _dataset = MSRVTTDataset
    elif dn == 'trecvid2018':
        _dataset = TRECVID2018Dataset
    elif dn == 'generic':
        _dataset = GenericDataset
    else:
        print("Invalid dataset specified...")
        sys.exit(1)

    dataset = _dataset(root=root, json_file=json_file, vocab=vocab,
                       subset=subset, transform=transform, skip_images=skip_images,
                       feature_loaders=feature_loaders)

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
    return data_loader


if __name__ == '__main__':
    # Test loading dataset!
    from pprint import pprint
    import pickle
    from build_vocab import Vocabulary

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
