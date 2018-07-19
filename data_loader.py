import torch
import torch.utils.data as data
import os
import nltk
import sys
import json
from PIL import Image


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_file, vocab, subset=None, transform=None):
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


class VisualGenomeIM2PDataset(data.Dataset):
    """Visual Genome / MS COCO Paragraph-length caption dataset"""

    def __init__(self, root, json_file, vocab, subset=None, transform=None):
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
    """VIST Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_file, vocab, subset=None, transform=None):
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

        self.captions = []

        for ann in self.anns:
            image_id = ann[0]['photo_flickr_id']
            if image_id in images:
                self.captions.append([image_id, ann[0]['text']])

        print("... {} images loaded ...".format(len(self.captions)))

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        image_id = self.captions[index][0]
        caption = self.captions[index][1]

        image_path = os.path.join(self.root, str(image_id) + '.jpg')
        if os.path.isfile(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            image_path = os.path.join(self.root, str(image_id) + '.png')
            image = Image.open(image_path).convert('RGB')

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


def get_loader(dataset_name, root, json_file, vocab, transform, batch_size,
               shuffle, num_workers, subset=None):
    """Returns torch.utils.data.DataLoader for user-specified dataset."""

    if dataset_name == 'coco':
        _dataset = CocoDataset
    elif dataset_name == 'vist':
        _dataset = VistDataset
    elif dataset_name == 'vgim2p':
        _dataset = VisualGenomeIM2PDataset
    else:
        print("Invalid dataset specified...")
        sys.exit(1)

    dataset = _dataset(root=root, json_file=json_file, vocab=vocab,
                       subset=subset, transform=transform)

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
                                              collate_fn=collate_fn)
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
