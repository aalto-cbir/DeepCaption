import argparse
import os
from PIL import Image
import shutil


def resize_image(image, size):
    """Resize an image to the given size."""
    try:
        return image.resize(size, Image.ANTIALIAS)
    except OSError as e:
        print("WARNING: unable to resize image {}: {}".format(image, str(e)))
        return None


def resize_images(image_dir, output_dir, create_zip, size):
    """Resize the images in 'image_dir' and save into 'output_dir'.
    'create_zip' tells whether we need to create a ZIP archive"""
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

    if create_zip:
        # Note that shutil.make_archive has issues with Python versions prior 3.5
        shutil.make_archive(output_dir, 'zip', os.path.dirname(output_dir),
                            os.path.basename(output_dir))


def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    create_zip = args.create_zip
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, create_zip, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        default='datasets/data/COCO/train2014',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str,
                        default='datasets/processed/COCO/train2014_resized',
                        help='directory for saving resized images')
    parser.add_argument('--create_zip', type=int, default=0,
                        help='set to 1 to create zip archive with resized images,'
                        'save ZIP file as "\{output_dir\}.zip"')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)
