import argparse
import os
from PIL import Image


def resize_image(image, size):
    """Resize an image to the given size."""
    try:
        return image.resize(size, Image.ANTIALIAS)
    except OSError as e:
        print("WARNING: unable to resize image {}: {}".format(image, str(e)))
        return None


def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, image)
        if os.path.isfile(output_path):
            print("{} exists, skipping...".format(output_path))
            continue
        with open(os.path.join(image_dir, image), 'rb') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                if img is not None:
                    img.save(output_path, img.format)
        if (i + 1) % 100 == 0:
            print("[{}/{}] Resized the images and saved into '{}'."
                  .format(i + 1, num_images, output_dir))


def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/train2014/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='./data/resized2014/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)
