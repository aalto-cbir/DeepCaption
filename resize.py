#!/usr/bin/env python3

import argparse
import os
import cv2
import shutil
import pymp
import multiprocessing
import glob


def resize_image(image, size):
    """Resize an image to the given size."""
    try:
        return cv2.resize(image, size)
    except OSError as e:
        print("WARNING: unable to resize image {}: {}".format(image, str(e)))
        return image


def save_image(output_path, img):
    # noinspection PyBroadException
    try:
        cv2.imwrite(output_path, img)
    except Exception as _:
        print('ERROR saving {}'.format(output_path))


def resize_images(image_dir, output_dir, create_zip, size):
    """Resize the images in 'image_dir' and save into 'output_dir'.
    'create_zip' tells whether we need to create a ZIP archive"""
    os.makedirs(output_dir, exist_ok=True)

    if args.subset:
        print('Resizing image subset defined in {} ...'.format(args.subset))
        subset_ids = [line.rstrip() for line in open(args.subset, 'r')]
        images = []
        for img_id in subset_ids:
            # Attempt to auto-detect file extension:
            img_ext = glob.glob(os.path.join(image_dir, img_id) + '.*')[0].split('.')[-1]
            images.append("{}.{}".format(img_id, img_ext))
    else:
        print('Resizing all images...')
        images = os.listdir(image_dir)
    num_images = len(images)

    # Run the job on several cores to speed it up:
    num_cores = min(4, multiprocessing.cpu_count())
    print('Using {} CPU cores'.format(num_cores))

    counter = pymp.shared.array((1,), dtype='uint32')
    with pymp.Parallel(num_cores) as p:
        for i in p.range(num_images):
            image = images[i]
            output_path = os.path.join(output_dir, image)
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                print("{} exists, skipping...".format(output_path))
            else:
                img = cv2.imread(os.path.join(image_dir, image))
                if img is not None:
                    img = resize_image(img, size)
                    save_image(output_path, img)
                else:
                    print('image {} not found or corrupted'.format(os.path.join(image_dir,
                                                                                image)))

            with p.lock:
                counter[0] += 1
                i = int(counter[0])
                if (i + 1) % 100 == 0:
                    print("[{}/{}] Resized the images and saved into '{}'."
                          .format(i + 1, num_images, output_dir))

    if create_zip:
        print("Creating a zip file: {}".format(output_dir + '.zip'))
        # Note that shutil.make_archive has issues with Python versions prior 3.5
        shutil.make_archive(output_dir, 'zip', os.path.dirname(output_dir),
                            os.path.basename(output_dir))


def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    create_zip = args.create_zip
    image_size = (args.image_size, args.image_size)
    resize_images(image_dir, output_dir, create_zip, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True,
                        help='directory for train images')
    parser.add_argument('--subset', type=str, default=None,
                        help='path to (optional) new-line separated file '
                             'listing ids of images to include')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='directory for saving resized images')
    parser.add_argument('--create_zip', action="store_true",
                        help='save ZIP file as "\{output_dir\}.zip"')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args=args)
