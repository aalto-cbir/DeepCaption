import argparse
import os
import cv2
import shutil
import pymp
import multiprocessing
import glob
from tqdm import tqdm


def resize_image(image, size):
    """Resize an image to the given size."""
    try:
        return cv2.resize(image, size)
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

    # Run the job on several cores to speed it up:
    num_cores = min(4, multiprocessing.cpu_count())
    print('Using {} CPU cores'.format(num_cores))

    # If no subset is specified:
    if not args.subset:
        print('Resizing all images...')
        with pymp.Parallel(num_cores) as p:
            for i in p.range(0, num_images):
                image = images[i]
                output_path = os.path.join(output_dir, image)
                if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                    print("{} exists, skipping...".format(output_path))
                    continue

                img = cv2.imread(os.path.join(image_dir, image))
                if img is not None:
                    img = resize_image(img, size)
                    cv2.imwrite(output_path, img)

                if (i + 1) % 100 == 0:
                    print("[{}/{}] Resized the images and saved into '{}'."
                          .format(i + 1, num_images, output_dir))
    # Load images defined as new-line separated subset of image-ids:
    else:
        print('Resizing image subset defined in {} ...'.format(args.subset))
        subset_ids = [line.rstrip() for line in open(args.subset)]
        num_images = len(subset_ids)
        pbar = tqdm(total=num_images)
        with pymp.Parallel(num_cores) as p:
            for i in p.range(0, num_images):
                img_id = subset_ids[i]
                # Attempt to auto-detect file extension:
                img_ext = glob.glob(os.path.join(image_dir, img_id) + '.*')[0].split('.')[-1]
                image = "{}.{}".format(img_id, img_ext)
                output_path = os.path.join(output_dir, image)
                if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                    print("{} exists, skipping...".format(output_path))
                    continue

                img = cv2.imread(os.path.join(image_dir, image))
                if img is not None:
                    img = resize_image(img, size)
                    cv2.imwrite(output_path, img)
                else:
                    print('image {} not found or corrupted'.format(os.path.join(image_dir, image)))

                #if (i + 1) % 100 == 0:
                #    print("[{}/{}] Resized the images and saved into '{}'."
                #          .format(i + 1, num_images, output_dir))
                pbar.update(1)



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
    parser.add_argument('--image_dir', type=str,
                        default='datasets/data/COCO/images/train2014',
                        help='directory for train images')
    parser.add_argument('--subset', type=str, default=None,
                        help='path to (optional) new-line separated file '
                        'listing ids of images to include')
    parser.add_argument('--output_dir', type=str,
                        default='datasets/processed/COCO/train2014_resized',
                        help='directory for saving resized images')
    parser.add_argument('--create_zip', action="store_true",
                        help='save ZIP file as "\{output_dir\}.zip"')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args=args)
