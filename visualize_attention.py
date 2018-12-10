import argparse
import json
import numpy as np
import skimage.transform
import os
import nltk
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Load captions, load numpy weights, plot image attention

# Resize image to square

# Resize attention mask to larger?


def visualize_att(image_path, caption, alphas, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from:
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/caption.py
    and https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param smooth: smooth weights?
    """
    num_locs = int(np.sqrt(alphas.shape[-1]))

    assert num_locs ** 2 == alphas.shape[-1], 'Alphas should be reshapable to square matrix'

    image = Image.open(image_path)
    image = image.resize([num_locs * 24, num_locs * 24], Image.LANCZOS)

    caption = nltk.tokenize.word_tokenize(str(caption).lower())

    for t in range(len(caption)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(caption) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (caption[t]), color='black', backgroundcolor='white',
                 fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :].reshape((num_locs, num_locs))
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, [num_locs * 24, num_locs * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    plt.show()


def main(args):
    # Load captions json
    with open(args.captions) as f:
        captions = json.load(f)

    # Load alphas numpy file
    alphas = np.load(args.alphas)

    assert len(captions) == len(alphas), 'Number of captions and alphas not matched'

    print("Starting to visualize attention:")

    for idx in args.indices:
        image_path = os.path.join(args.images_root, captions[idx]['image_id'])
        image_caption = captions[idx]['caption']
        image_alphas = alphas[idx]

        print("=" * 90)
        print("Image: {}".format(image_path))
        print("Caption: {}".format(image_caption))

        visualize_att(image_path, image_caption, image_alphas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indices', type=int, nargs='*',
                        help='Indices inside caption json file and alphas '
                        'file corresponding to the images we want look at')
    parser.add_argument('--captions', type=str,
                        help='path to JSON file with generated captions')
    parser.add_argument('--alphas', type=str,
                        help='path to NumPy file containing attention weights')
    parser.add_argument('--images_root', type=str,
                        help='directory where images can be found')

    args = parser.parse_args()
    main(args=args)
