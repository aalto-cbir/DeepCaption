import torch
import torch.utils.data as data
import os
import nltk
import sys
import json
from PIL import Image


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_file, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json_file: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(json_file)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        print("... {} images loaded ...".format(len(self.ids)))

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


class VistDataset(data.Dataset):
    """VIST Custom Dataset for sequence processing, compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_file, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json_file: VIST annotation file path.
            vocab: vocabulary wrapper.
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

        print("... {} sequences loaded ...".format(len(self.data_hold)))

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
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
        return sequence, target

    def __len__(self):
        return len(self.data_hold)


class MSRVTTDataset(data.Dataset):
    """MSR-VTT Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_file, vocab, transform=None):
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

        print("... {} images, {} captions loaded ...".format(
            len(train_vids), len(self.captions)))

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        # ann_id = self.ids[index]
        # caption = coco.anns[ann_id]['caption']
        # img_id = coco.anns[ann_id]['image_id']
        # path = coco.loadImgs(img_id)[0]['file_name']

        vid = self.captions[index][0]
        caption = self.captions[index][1]

        assert vid[:5] == 'video'
        vid_idx = int(vid[5:])
        path = '{:04}:kf1.jpeg'.format(vid_idx)

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        vocab = self.vocab
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.captions)


def collate_fn(data):
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
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


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
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    # images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(dataset_name, root, json_file, vocab, transform, batch_size,
               shuffle, num_workers, _collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for user-specified dataset."""

    dn = dataset_name.lower()

    if dn == 'coco':
        _dataset = CocoDataset
    elif 'vist' in dn:
        _dataset = VistDataset
    elif dn == 'msrvtt' or dn == 'msr-vtt':
        _dataset = MSRVTTDataset
    else:
        print("Invalid dataset specified...")
        sys.exit(1)

    dataset = _dataset(root=root, json_file=json_file, vocab=vocab, transform=transform)

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
